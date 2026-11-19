# Push this project to GitHub

This folder is already a Git repository on branch `main`. Create an empty public repository under your GitHub account first. Do not initialize it with a README, license, or `.gitignore`.

In PowerShell, from this folder, run:

```powershell
git remote add origin https://github.com/shashankk0304-art/YOUR-REPOSITORY-NAME.git
git push -u origin main
```

Replace `YOUR-REPOSITORY-NAME` with the repository name you created. Git will ask you to authenticate using your configured Git Credential Manager or browser sign-in; never paste a password or token into this document.

If you are pushing from another computer, use the companion `EcoSort_ARM_Submission.bundle` file:

```powershell
git clone .\EcoSort_ARM_Submission.bundle EcoSort_ARM_Submission
Set-Location .\EcoSort_ARM_Submission
git remote add origin https://github.com/shashankk0304-art/YOUR-REPOSITORY-NAME.git
git push -u origin main
```

The bundle contains the committed files and history. It excludes ignored data caches, local keys, Python caches, and the redundant `kv260_multilingual_update.tar.gz` transfer archive. The Llama ExecuTorch PTE, tokenizer, and `llama_main` runtime are board-side prerequisites and are documented separately in `RUN_ON_KV260.md`.
