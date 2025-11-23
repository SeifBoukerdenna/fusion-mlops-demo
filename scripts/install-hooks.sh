#!/bin/bash
# Install git pre-commit hook

echo "Installing pre-commit hook..."

cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Auto-run validation before commit

echo "Running pre-commit checks..."
./scripts/pre-commit-check.sh

exit $?
EOF

chmod +x .git/hooks/pre-commit

echo "✓ Pre-commit hook installed!"
echo "It will now run automatically before every commit"