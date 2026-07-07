#!/data/data/com.termux/files/usr/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Audio Processing Pipeline — Termux installer
# Run once after cloning the repo:
#   bash termux_install.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RESET="\033[0m"

info()    { echo -e "${GREEN}[✓]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[!]${RESET} $*"; }
section() { echo -e "\n${BOLD}── $* ──${RESET}"; }

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║  Audio Processing Pipeline — Termux      ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${RESET}"

# ── 1. System packages ────────────────────────────────────────────────────────
section "Updating Termux packages"
pkg update -y -o Dpkg::Options::="--force-confnew" 2>/dev/null || pkg update -y
info "Package index updated"

section "Installing system dependencies"
pkg install -y python ffmpeg libsndfile openssl 2>/dev/null
info "python, ffmpeg, libsndfile, openssl installed"

# ── 2. Python packages ────────────────────────────────────────────────────────
section "Installing Python packages"
pip install --upgrade pip --quiet

# Core packages — always installed
pip install \
    faster-whisper \
    onnxruntime \
    scipy \
    numpy \
    --quiet
info "faster-whisper, onnxruntime, scipy, numpy installed"

# Optional: LLM features (post-correction, diarization, name recognition)
echo ""
read -r -p "Install OpenAI package for LLM features (--post-correct, --name-speakers)? [y/N] " INSTALL_OPENAI
if [[ "$INSTALL_OPENAI" =~ ^[Yy]$ ]]; then
    pip install openai --quiet
    info "openai installed"
    echo ""
    warn "Set your API key before using LLM features:"
    warn "  export OPENAI_API_KEY=sk-..."
    warn "  Add this line to ~/.bashrc to make it permanent."
fi

# ── 3. Create convenience scripts ─────────────────────────────────────────────
section "Creating convenience scripts"

cat > process.sh << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
# Process a single audio file with the full pipeline.
# Usage: bash process.sh <audio_file> [extra flags]
#
# Examples:
#   bash process.sh call.amr
#   bash process.sh call.amr --studio --post-correct
#   bash process.sh call.amr --studio --name-speakers --keep-timing

cd "$(dirname "$0")"

if [ -z "$1" ]; then
    echo "Usage: bash process.sh <audio_file> [flags]"
    echo ""
    echo "Common flags:"
    echo "  --studio          Studio-quality audio enhancement"
    echo "  --post-correct    LLM post-correction (requires OPENAI_API_KEY)"
    echo "  --name-speakers   Identify speaker names from transcript"
    echo "  --keep-timing     Keep original timing in per-speaker files"
    echo "  --vad-threshold   Voice activity detection (e.g. -30)"
    echo "  --model           Whisper model: tiny, base, small, medium, large-v3"
    echo ""
    exit 1
fi

FILE="$1"
shift

echo ""
echo "  Processing: $FILE"
echo "  Flags: $*"
echo ""

python AudioPipeline.py process "$FILE" "$@"
EOF
chmod +x process.sh
info "process.sh created"

cat > watch.sh << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
# Watch a folder and auto-process any audio file dropped into it.
# Usage: bash watch.sh [--inbox ./inbox] [extra flags]

cd "$(dirname "$0")"

INBOX="${INBOX:-./inbox}"
OUTPUT="${OUTPUT:-./output}"

mkdir -p "$INBOX" "$OUTPUT"

echo ""
echo "  Watching: $INBOX"
echo "  Output:   $OUTPUT"
echo "  Drop audio files into the inbox folder to process them automatically."
echo "  Press Ctrl+C to stop."
echo ""

python AudioPipeline.py watch \
    --inbox "$INBOX" \
    --output "$OUTPUT" \
    --studio \
    "$@"
EOF
chmod +x watch.sh
info "watch.sh created"

# ── 4. Done ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}Installation complete!${RESET}"
echo ""
echo "  Process a file:     bash process.sh call.amr --studio"
echo "  Watch a folder:     bash watch.sh"
echo "  Full CLI help:      python AudioPipeline.py --help"
echo ""
echo "  See docs/TERMUX.md for detailed usage and tips."
echo ""
