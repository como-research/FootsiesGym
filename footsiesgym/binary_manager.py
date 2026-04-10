"""
Binary manager for FootsiesGym - handles automatic binary downloads.
"""

import hashlib
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    import msvcrt

    def _flock(fd):
        """Acquire an exclusive lock on the file descriptor (Windows)."""
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)

else:
    import fcntl

    def _flock(fd):
        """Acquire an exclusive lock on the file descriptor (Unix)."""
        fcntl.flock(fd, fcntl.LOCK_EX)


class BinaryManager:
    """Manages automatic downloading and caching of FootsiesGym binaries."""

    # Primary download URL (Cloudflare R2 bucket)
    # NOTE(chase): This must be updated with each release to point to the correct versioned directory.
    # It's OK if it points to an old version as long as it's OK the the binaries are identical
    # and the interface with them haven't changed.
    DOWNLOAD_BASE_URL = "https://footsiesgym.chasemcd.com/v0.7.0"

    # Fallback URLs in case primary fails
    FALLBACK_URLS = [
        "https://github.com/chasemcd/FootsiesGym/raw/main/binaries",
        "https://raw.githubusercontent.com/chasemcd/FootsiesGym/main/binaries",
    ]

    BINARY_FILES = {
        "footsies_linux_headless_9c6b36f.zip": "footsies_linux_headless_9c6b36f.zip",
        "footsies_linux_windowed_9c6b36f.zip": "footsies_linux_windowed_9c6b36f.zip",
        "footsies_mac_headless_5709b6d.zip": "footsies_mac_headless_5709b6d.zip",
        "footsies_mac_windowed_5709b6d.zip": "footsies_mac_windowed_5709b6d.zip",
    }

    # SHA256 hashes for integrity verification
    BINARY_HASHES = {
        "footsies_linux_headless_9c6b36f.zip": "1224a165bce03272e4a01809ed00280cb61b9d0426500e8eb5a72d5692b0a5d1",
        "footsies_linux_windowed_9c6b36f.zip": "c0b6d3510b9f498dabdb93bbe2abee9dfe3da7e62b657a6fa732f0efbfcd5cce",
        "footsies_mac_headless_5709b6d.zip": "7d4c931a7ace0fa34d518959713e4add7e44e4e0f128c62e8b57a9b5a052104a",
        "footsies_mac_windowed_5709b6d.zip": "658b3b3fc37e4e5cfb92f72799aa22ce92320957b2cac668fc7a2765deb45073",
    }

    def __init__(self):
        self.package_dir = Path(__file__).parent
        self.binaries_dir = self.package_dir / "binaries"
        self.binaries_dir.mkdir(exist_ok=True)

    def ensure_binaries_available(self, platform: str = "linux") -> bool:
        """
        Ensure the required binaries are available for the given platform.
        Downloads them automatically if they don't exist locally.
        Uses file locking to prevent race conditions when multiple processes
        try to download simultaneously.

        Args:
            platform: Target platform ("linux" or "mac")

        Returns:
            bool: True if binaries are available, False otherwise
        """
        required_files = self._get_required_files(platform)

        # Check if all required files exist
        missing_files = []
        for filename in required_files:
            file_path = self.binaries_dir / filename
            if not file_path.exists():
                missing_files.append(filename)

        if not missing_files:
            print(f"✅ All {platform} binaries are available")
            return True

        # Use file locking to prevent multiple processes from downloading simultaneously
        lock_file = self.binaries_dir / ".download_lock"

        try:
            # Create lock file and acquire exclusive lock
            with open(lock_file, "w") as lock_fd:
                print(f"🔒 Acquiring download lock for {platform} binaries...")
                _flock(lock_fd.fileno())

                # Re-check if files exist after acquiring lock (another process might have downloaded them)
                missing_files = []
                for filename in required_files:
                    file_path = self.binaries_dir / filename
                    if not file_path.exists():
                        missing_files.append(filename)

                if not missing_files:
                    print(
                        f"✅ All {platform} binaries are now available (downloaded by another process)"
                    )
                    return True

                print(
                    f"📥 Downloading missing {platform} binaries: {missing_files}"
                )

                # Download missing files
                success = True
                for filename in missing_files:
                    if not self._download_binary(filename):
                        success = False

                print(f"🔓 Releasing download lock for {platform} binaries")
                return success

        except Exception as e:
            print(f"❌ Error during binary download: {e}")
            return False
        finally:
            # Clean up lock file
            try:
                if lock_file.exists():
                    lock_file.unlink()
            except:
                pass  # Ignore cleanup errors

    def ensure_binaries_extracted(
        self,
        platform: str = "linux",
        target_dir: str = None,
        headless: bool = True,
    ) -> bool:
        """
        Ensure binaries are downloaded and extracted to the target directory.
        Uses file locking to prevent race conditions during both download and extraction.

        Args:
            platform: Target platform ("linux" or "mac")
            target_dir: Directory to extract binaries to
            headless: Whether to use headless binary (True) or windowed (False)

        Returns:
            bool: True if binaries are available and extracted, False otherwise
        """
        if not target_dir:
            target_dir = str(self.package_dir.parent / "binaries")

        # Use separate directories for headless vs windowed binaries
        binary_subdir = (
            "footsies_binaries_headless"
            if headless
            else "footsies_binaries_windowed"
        )
        binary_path = os.path.join(
            target_dir, binary_subdir, "footsies.x86_64"
        )

        # Check if extracted binaries already exist
        if os.path.exists(binary_path):
            binary_type = "headless" if headless else "windowed"
            print(
                f"✅ Extracted {platform} {binary_type} binaries already available at {target_dir}"
            )
            return True

        # Use file locking to prevent multiple processes from downloading/extracting simultaneously
        lock_file = self.binaries_dir / ".extraction_lock"

        try:
            # Create lock file and acquire exclusive lock
            with open(lock_file, "w") as lock_fd:
                print(
                    f"🔒 Acquiring extraction lock for {platform} binaries..."
                )
                _flock(lock_fd.fileno())

                # Re-check if extracted binaries exist after acquiring lock
                if os.path.exists(binary_path):
                    binary_type = "headless" if headless else "windowed"
                    print(
                        f"✅ Extracted {platform} {binary_type} binaries now available (extracted by another process)"
                    )
                    return True

                # First ensure the zip files are downloaded
                if not self.ensure_binaries_available(platform):
                    print(f"❌ Failed to download {platform} binaries")
                    return False

                # Get the appropriate zip file
                if platform.lower() == "linux":
                    zip_filename = (
                        "footsies_linux_headless_9c6b36f.zip"
                        if headless
                        else "footsies_linux_windowed_9c6b36f.zip"
                    )
                else:
                    zip_filename = (
                        "footsies_mac_headless_5709b6d.zip"
                        if headless
                        else "footsies_mac_windowed_5709b6d.zip"
                    )

                zip_path = self.binaries_dir / zip_filename
                if not zip_path.exists():
                    print(
                        f"❌ Zip file {zip_filename} not found after download"
                    )
                    return False

                print(f"📦 Extracting {zip_filename} to {target_dir}...")

                # Create target directory
                os.makedirs(target_dir, exist_ok=True)

                # Extract the zip file
                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(target_dir)
                except Exception as e:
                    print(f"❌ Failed to extract {zip_filename}: {e}")
                    return False

                # Find the extracted folder and rename it to footsies_binaries
                extracted_items = os.listdir(target_dir)
                extracted_folder = None

                for item in extracted_items:
                    item_path = os.path.join(target_dir, item)
                    if (
                        os.path.isdir(item_path)
                        and item != binary_subdir
                        and not item.endswith(".zip")
                        and "footsies" in item.lower()
                    ):
                        extracted_folder = item
                        break

                if extracted_folder:
                    old_path = os.path.join(target_dir, extracted_folder)
                    new_path = os.path.join(target_dir, binary_subdir)

                    # Remove existing directory if it exists
                    if os.path.exists(new_path):
                        shutil.rmtree(new_path)

                    os.rename(old_path, new_path)
                    binary_type = "headless" if headless else "windowed"
                    print(
                        f"📁 Renamed {extracted_folder} to {binary_subdir} ({binary_type})"
                    )

                # Verify the binary now exists
                if not os.path.exists(binary_path):
                    print(
                        f"❌ Failed to find binary at {binary_path} after extraction"
                    )
                    return False

                # Make binary executable
                os.chmod(binary_path, 0o755)

                print(f"🔓 Releasing extraction lock for {platform} binaries")
                return True

        except Exception as e:
            print(f"❌ Error during binary extraction: {e}")
            return False
        finally:
            # Clean up lock file
            try:
                if lock_file.exists():
                    lock_file.unlink()
            except:
                pass  # Ignore cleanup errors

    def _get_required_files(self, platform: str) -> list[str]:
        """Get the list of required binary files for a platform."""
        if platform.lower() == "linux":
            return [
                "footsies_linux_headless_9c6b36f.zip",
                "footsies_linux_windowed_9c6b36f.zip",
            ]
        elif platform.lower() == "mac":
            return [
                "footsies_mac_headless_5709b6d.zip",
                "footsies_mac_windowed_5709b6d.zip",
            ]
        else:
            return []

    def _verify_hash(self, filename: str) -> bool:
        """
        Verify the SHA256 hash of a downloaded binary file.

        Args:
            filename: Name of the file to verify

        Returns:
            bool: True if hash matches or no hash is registered, False on mismatch
        """
        expected_hash = self.BINARY_HASHES.get(filename)
        if expected_hash is None:
            return True

        file_path = self.binaries_dir / filename
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                sha256.update(chunk)

        actual_hash = sha256.hexdigest()
        if actual_hash != expected_hash:
            print(
                f"  ❌ Hash mismatch for {filename}:\n"
                f"     expected {expected_hash}\n"
                f"     got      {actual_hash}"
            )
            file_path.unlink(missing_ok=True)
            return False

        print(f"  ✅ Hash verified for {filename}")
        return True

    def _download_binary(self, filename: str) -> bool:
        """
        Get a binary file by copying from local repository or downloading from HTTP.

        Args:
            filename: Name of the file to download

        Returns:
            bool: True if successful, False otherwise
        """
        local_path = self.binaries_dir / filename

        # First, try to find the file in the local repository
        # Look for it in the parent directory structure
        repo_binaries_path = (
            self.package_dir.parent / "footsiesgym" / "binaries" / filename
        )

        if repo_binaries_path.exists():
            try:
                print(f"  Copying {filename} from local repository...")
                import shutil

                shutil.copy2(repo_binaries_path, local_path)
                print(
                    f"  ✅ Copied {filename} ({local_path.stat().st_size} bytes)"
                )
                if not self._verify_hash(filename):
                    return False
                return True
            except Exception as e:
                print(f"  ❌ Failed to copy from local repository: {e}")

        # If local copy fails, try HTTP download as fallback
        urls_to_try = [self.DOWNLOAD_BASE_URL] + self.FALLBACK_URLS

        for base_url in urls_to_try:
            url = f"{base_url}/{filename}"

            try:
                print(f"  Downloading {filename} from {base_url}...")

                # Create request with user agent to avoid blocking
                request = urllib.request.Request(url)
                request.add_header("User-Agent", "FootsiesGym/0.7.1")

                with urllib.request.urlopen(request, timeout=30) as response:
                    # Check if we got a valid response
                    if response.status == 200:
                        with open(local_path, "wb") as f:
                            # Download in chunks to handle large files
                            while True:
                                chunk = response.read(8192)
                                if not chunk:
                                    break
                                f.write(chunk)

                        print(
                            f"  ✅ Downloaded {filename} ({local_path.stat().st_size} bytes)"
                        )
                        if not self._verify_hash(filename):
                            continue
                        return True
                    else:
                        print(f"  ❌ HTTP {response.status} from {base_url}")
                        continue

            except urllib.error.HTTPError as e:
                print(f"  ❌ HTTP error {e.code} from {base_url}: {e.reason}")
                continue
            except urllib.error.URLError as e:
                print(f"  ❌ URL error from {base_url}: {e.reason}")
                continue
            except Exception as e:
                print(f"  ❌ Unexpected error from {base_url}: {e}")
                continue

        print(f"  ❌ Failed to get {filename} from all sources")
        return False

    def get_binary_path(
        self, platform: str = "linux", windowed: bool = False
    ) -> Optional[Path]:
        """
        Get the path to the appropriate binary file.

        Args:
            platform: Target platform ("linux" or "mac")
            windowed: Whether to get windowed version (True) or headless (False)

        Returns:
            Path to the binary file, or None if not available
        """
        if platform.lower() == "linux":
            filename = (
                "footsies_linux_windowed_9c6b36f.zip"
                if windowed
                else "footsies_linux_headless_9c6b36f.zip"
            )
        elif platform.lower() == "mac":
            filename = (
                "footsies_mac_windowed_5709b6d.zip"
                if windowed
                else "footsies_mac_headless_5709b6d.zip"
            )
        else:
            return None

        binary_path = self.binaries_dir / filename
        return binary_path if binary_path.exists() else None


# Global instance
_binary_manager = None


def get_binary_manager() -> BinaryManager:
    """Get the global binary manager instance."""
    global _binary_manager
    if _binary_manager is None:
        _binary_manager = BinaryManager()
    return _binary_manager
