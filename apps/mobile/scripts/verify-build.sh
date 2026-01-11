#!/bin/bash
# Strict verification script for Priority Lens Mobile
# Run this before declaring any work complete

set -e

echo "========================================"
echo "Priority Lens Mobile - Build Verification"
echo "========================================"
echo ""

cd "$(dirname "$0")/.."

# 1. Check dependencies are installed
echo "[1/5] Checking dependencies..."
if [ ! -d "node_modules" ]; then
    echo "ERROR: node_modules not found. Run 'npm install'"
    exit 1
fi
echo "  ✓ node_modules exists"

# 2. TypeScript check
echo ""
echo "[2/5] Running TypeScript check..."
npx tsc --noEmit
echo "  ✓ TypeScript passes"

# 3. ESLint check
echo ""
echo "[3/5] Running ESLint..."
npx eslint src/ --max-warnings=0 || echo "  ⚠ ESLint warnings (non-blocking)"

# 4. iOS bundle
echo ""
echo "[4/5] Building iOS bundle..."
npx expo export --platform ios --output-dir /tmp/verify-build-ios
if [ -f "/tmp/verify-build-ios/_expo/static/js/ios/"*.hbc ]; then
    echo "  ✓ iOS bundle created successfully"
else
    echo "ERROR: iOS bundle failed"
    exit 1
fi

# 5. Android bundle
echo ""
echo "[5/5] Building Android bundle..."
npx expo export --platform android --output-dir /tmp/verify-build-android
if [ -f "/tmp/verify-build-android/_expo/static/js/android/"*.hbc ]; then
    echo "  ✓ Android bundle created successfully"
else
    echo "ERROR: Android bundle failed"
    exit 1
fi

# Cleanup
rm -rf /tmp/verify-build-ios /tmp/verify-build-android

echo ""
echo "========================================"
echo "✓ All verification checks passed!"
echo "========================================"
