"""Generate the offline demo recording with local Piper/VITS voices.

This produces TEST INPUT only. Nothing here fabricates transcription or minutes:
the generated WAV is fed through the same local Whisper/diarization/LLM pipeline
an evaluator runs on their own recordings. Requires the optional TTS voices from
`scripts/download_models.py --demo-voices`.
"""

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SILENCE_SECONDS = 0.45

# (voice directory, spoken line). Russian, Kazakh and one code-switched line.
SCRIPT = [
    ("ru_RU-denis-medium", "Коллеги, начинаем планёрку по проекту Алем."),
    ("ru_RU-denis-medium", "Первый вопрос — отчёт по продажам за сентябрь."),
    (
        "ru_RU-irina-medium",
        "Я подготовлю отчёт по продажам к двадцать пятому сентября две тысячи "
        "двадцать шестого года.",
    ),
    ("ru_RU-denis-medium", "Хорошо. Ерлан, свяжись с подрядчиком до конца недели."),
    ("kk_KZ-issai-high", "Жарайды, мен мердігермен осы аптаның соңына дейін байланысамын."),
    ("kk_KZ-issai-high", "Сондай-ақ мен бюджет есебін дайындаймын."),
    ("ru_RU-irina-medium", "И ещё, мен презентацияны дайындаймын, отправлю всем в понедельник."),
    ("ru_RU-denis-medium", "Отлично, на этом заканчиваем."),
]


def engine(folder: Path):
    import sherpa_onnx

    name = folder.name.removeprefix("vits-piper-")
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(folder / f"{name}.onnx"),
                tokens=str(folder / "tokens.txt"),
                data_dir=str(folder / "espeak-ng-data"),
            ),
            provider="cpu",
            num_threads=4,
        ),
        max_num_sentences=1,
    )
    if not config.validate():
        raise ValueError(f"Invalid TTS configuration for {folder}")
    return sherpa_onnx.OfflineTts(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models/tts"))
    parser.add_argument("--output", type=Path, default=Path("assets/demo/planning-ru-kk.wav"))
    args = parser.parse_args()

    engines, rate, chunks = {}, None, []
    for voice, text in SCRIPT:
        folder = args.model_dir / f"vits-piper-{voice}"
        if not folder.is_dir():
            raise SystemExit(
                f"Missing voice {folder}. Run: python scripts/download_models.py --demo-voices"
            )
        if voice not in engines:
            engines[voice] = engine(folder)
        audio = engines[voice].generate(text, sid=0, speed=1.0)
        if rate is None:
            rate = audio.sample_rate
        elif rate != audio.sample_rate:
            raise SystemExit(f"Voice sample rates differ: {rate} vs {audio.sample_rate}")
        chunks.append(np.asarray(audio.samples, dtype=np.float32))
        chunks.append(np.zeros(int(rate * SILENCE_SECONDS), dtype=np.float32))
        print(f"{voice}: {text}", flush=True)

    samples = np.concatenate(chunks)
    peak = float(np.max(np.abs(samples))) or 1.0
    pcm = (samples / peak * 0.9 * 32767).astype("<i2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    print(f"Demo audio: {args.output} ({len(samples) / rate:.1f}s, {rate} Hz)")


if __name__ == "__main__":
    main()
