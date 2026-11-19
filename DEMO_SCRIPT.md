# EcoSort demonstration recording checklist

The challenge requires a demonstration video. No video file was present in the supplied project folder, so record one after you verify the final build.

Suggested 3–4 minute flow:

1. Show the KV260 health response at `/api/health`; explain that the web app, OpenCV models, and ExecuTorch Llama run locally on the board.
2. Upload a clear image of a common recyclable item and show detection, disposal guidance, recycling options, and the environmental learning section.
3. Show the chatbot answering a waste question. State that Llama creates the short advice and the knowledge base supplies grounded reasoning and learning facts.
4. Change the language selector and show localized interface and answer behavior. English and Hindi microphone recognition run with on-board Vosk; Telugu, Kannada, and Marathi use browser recognition and may require internet. Demonstrate output speech only for a language voice installed on the browser device.
5. If laptop live camera is demonstrated, use the PuTTY localhost SSH tunnel described in `README.md` and grant camera permission in the browser.
6. Close with the measured CPU benchmark and limitations in `REPORT.md`. The active 30-class detector's can/earphone confusion is documented; do not present the measured smoke checks as general accuracy.

Recording notes:

- Keep the demo consistent with `REPORT.md`: CV and Llama currently run on the Arm CPU/PS. Do not claim FPGA/PL acceleration; the inspected board had no active DPU/XRT device.
- Do not show SSH private keys, passwords, shell history with credentials, or model files whose license terms are not cleared.
- Capture the final video as `demo/EcoSort_KV260_Demo.mp4` after recording. That path is ignored only if you add it to `.gitignore`; check the challenge's preferred video hosting/size rules before committing a large video to Git.
