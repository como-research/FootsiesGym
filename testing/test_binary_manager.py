"""Tests for binary_manager.py - URL updates and hash verification."""

import hashlib
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from footsiesgym import __version__
from footsiesgym.binary_manager import BinaryManager


@pytest.fixture
def manager(tmp_path):
    """Create a BinaryManager with a temporary binaries directory."""
    mgr = BinaryManager()
    mgr.binaries_dir = tmp_path / "binaries"
    mgr.binaries_dir.mkdir()
    return mgr


@pytest.fixture
def real_mac_zips():
    """Paths to the real mac binary zips in the repo."""
    repo_root = Path(__file__).parent.parent
    binaries_dir = repo_root / "binaries"
    return {
        "headless": binaries_dir / "footsies_mac_headless_5709b6d.zip",
        "windowed": binaries_dir / "footsies_mac_windowed_5709b6d.zip",
    }


class TestDownloadURL:
    def test_primary_url_points_to_cloudflare(self):
        expected_url = f"https://footsiesgym.chasemcd.com/v{__version__}"
        assert BinaryManager.DOWNLOAD_BASE_URL == expected_url

    def test_github_urls_are_fallbacks(self):
        assert len(BinaryManager.FALLBACK_URLS) == 2
        assert any("github.com" in url for url in BinaryManager.FALLBACK_URLS)
        assert any(
            "raw.githubusercontent.com" in url
            for url in BinaryManager.FALLBACK_URLS
        )

    def test_all_binary_files_have_hashes(self):
        for filename in BinaryManager.BINARY_FILES:
            assert (
                filename in BinaryManager.BINARY_HASHES
            ), f"Missing hash for {filename}"


class TestHashVerification:
    def test_verify_valid_file(self, manager, real_mac_zips):
        """Verify hash passes for the real mac headless binary."""
        src = real_mac_zips["headless"]
        if not src.exists():
            pytest.skip("Mac headless binary not found in repo")

        filename = "footsies_mac_headless_5709b6d.zip"
        shutil.copy2(src, manager.binaries_dir / filename)

        assert manager._verify_hash(filename) is True

    def test_verify_valid_file_windowed(self, manager, real_mac_zips):
        """Verify hash passes for the real mac windowed binary."""
        src = real_mac_zips["windowed"]
        if not src.exists():
            pytest.skip("Mac windowed binary not found in repo")

        filename = "footsies_mac_windowed_5709b6d.zip"
        shutil.copy2(src, manager.binaries_dir / filename)

        assert manager._verify_hash(filename) is True

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

    def test_stored_hashes_match_real_binaries(self, real_mac_zips):
        """Validate that BINARY_HASHES match the actual files on disk."""
        for key, path in real_mac_zips.items():
            if not path.exists():
                pytest.skip(f"Binary {path.name} not found in repo")

            sha256 = hashlib.sha256()
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    sha256.update(chunk)

            expected = BinaryManager.BINARY_HASHES[path.name]
            assert (
                sha256.hexdigest() == expected
            ), f"Hash mismatch for {path.name}"


class TestDownloadWithHashCheck:
    def test_local_copy_with_valid_hash(self, manager, real_mac_zips):
        """_download_binary should succeed when local copy has valid hash."""
        src = real_mac_zips["headless"]
        if not src.exists():
            pytest.skip("Mac headless binary not found in repo")

        filename = "footsies_mac_headless_5709b6d.zip"

        # Point the local repo lookup to actual binaries
        repo_binaries = manager.package_dir.parent / "footsiesgym" / "binaries"
        with patch.object(manager, "package_dir", manager.package_dir):
            # Manually set up the path so local copy works
            local_src = (
                manager.binaries_dir.parent / "footsiesgym" / "binaries"
            )
            local_src.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, local_src / filename)
            manager.package_dir = manager.binaries_dir.parent

            result = manager._download_binary(filename)

        assert result is True
        assert (manager.binaries_dir / filename).exists()

    def test_local_copy_with_bad_hash_fails(self, manager):
        """_download_binary should fail when local copy has wrong hash."""
        filename = "footsies_mac_headless_5709b6d.zip"

        # Create a fake local repo with a bad file
        local_src = manager.binaries_dir.parent / "footsiesgym" / "binaries"
        local_src.mkdir(parents=True, exist_ok=True)
        (local_src / filename).write_bytes(b"corrupted data")
        manager.package_dir = manager.binaries_dir.parent

        # Mock HTTP so it doesn't actually try to download
        with patch.object(manager, "DOWNLOAD_BASE_URL", "http://invalid.test"):
            with patch.object(manager, "FALLBACK_URLS", []):
                result = manager._download_binary(filename)

        assert result is False
        assert not (manager.binaries_dir / filename).exists()


class TestCloudflareDownload:
    def test_download_mac_headless_from_cloudflare(self, manager):
        """Integration test: download mac headless binary from Cloudflare and verify hash."""
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
