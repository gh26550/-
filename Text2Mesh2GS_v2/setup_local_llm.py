"""Install a user-local Ollama runtime on Linux x86_64; no system service or sudo."""
from pathlib import Path
import platform
import tarfile
import urllib.request
import zstandard


def main():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Run this installer inside x86_64 WSL/Linux")
    root = Path.home() / ".local/share/text2mesh2gs"
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "ollama"
    if (dest / "bin/ollama").exists():
        print("Ollama already installed: " + str(dest))
        return
    archive = root / "ollama-download.tar.zst"
    urllib.request.urlretrieve("https://ollama.com/download/ollama-linux-amd64.tar.zst", archive)
    dest.mkdir(exist_ok=True)
    with archive.open("rb") as stream, zstandard.ZstdDecompressor().stream_reader(stream) as decoded:
        with tarfile.open(fileobj=decoded, mode="r|") as package:
            package.extractall(dest, filter="data")
    print("Installed. Run bash serve_local_llm.sh, then in another terminal:")
    print(str(dest / "bin/ollama") + " pull qwen2.5vl:7b")


if __name__ == "__main__":
    main()
