"""Tests for binary_manager.py - download URLs and hash verification."""

import hashlib

import pytest

from footsiesgym.binary_manager import BinaryManager


@pytest.fixture
def manager(tmp_path):
    """Create a BinaryManager with a temporary binaries directory."""
    mgr = BinaryManager()
    mgr.binaries_dir = tmp_path / "binaries"
    mgr.binaries_dir.mkdir()
    return mgr


class TestDownloadURL:
    def test_primary_url_points_to_cloudflare(self):
        assert (
            BinaryManager.DOWNLOAD_BASE_URL == "https://footsiesgym.chasemcd.com/v0.7.0"
        )

    def test_github_releases_are_fallbacks(self):
        assert len(BinaryManager.FALLBACK_URLS) >= 1
        assert all(
            "github.com/como-research/FootsiesGym/releases" in url
            for url in BinaryManager.FALLBACK_URLS
        )

    def test_all_binary_files_have_hashes(self):
        for filename in BinaryManager.BINARY_FILES:
            assert (
                filename in BinaryManager.BINARY_HASHES
            ), f"Missing hash for {filename}"


class TestHashVerification:
    def test_verify_corrupted_file(self, manager):
        """Verify hash fails and deletes file when content is wrong."""
        filename = "footsies_mac_headless_5709b6d.zip"
        bad_file = manager.binaries_dir / filename
        bad_file.write_bytes(b"this is not a real binary")

        assert manager._verify_hash(filename) is False
        assert not bad_file.exists(), "Corrupted file should be deleted"

    def test_verify_unknown_file_passes(self, manager):
        """Files without a registered hash should pass verification."""
        filename = "unknown_file.zip"
        (manager.binaries_dir / filename).write_bytes(b"anything")

        assert manager._verify_hash(filename) is True

    def test_download_with_bad_sources_fails(self, manager, monkeypatch):
        """_download_binary should fail cleanly when no source is reachable."""
        monkeypatch.setattr(manager, "DOWNLOAD_BASE_URL", "http://invalid.test")
        monkeypatch.setattr(manager, "FALLBACK_URLS", [])

        filename = "footsies_mac_headless_5709b6d.zip"
        assert manager._download_binary(filename) is False
        assert not (manager.binaries_dir / filename).exists()


class TestCloudflareDownload:
    @pytest.mark.slow
    def test_download_mac_headless_from_cloudflare(self, manager):
        """Integration test: download mac headless binary and verify hash."""
        filename = "footsies_mac_headless_5709b6d.zip"
        result = manager._download_binary(filename)

        assert result is True
        assert (manager.binaries_dir / filename).exists()

        # Double-check the hash independently
        sha256 = hashlib.sha256()
        with open(manager.binaries_dir / filename, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                sha256.update(chunk)
        assert sha256.hexdigest() == BinaryManager.BINARY_HASHES[filename]
