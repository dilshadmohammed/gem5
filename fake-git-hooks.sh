mkdir -p .git/hooks
echo -e "#\!/bin/sh\nexit 0" > .git/hooks/pre-commit
echo -e "#\!/bin/sh\nexit 0" > .git/hooks/commit-msg
chmod +x .git/hooks/pre-commit .git/hooks/commit-msg
