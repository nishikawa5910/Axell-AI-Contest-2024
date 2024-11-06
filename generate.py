import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# 画像を読み込み、ONNXモデルとBicubic補間で解像度を上げた画像を保存し、PSNRを計算
def upscale_image_and_calculate_psnr(input_image_path, output_dir, model_path, upscale_factor=4):
    # 元画像（高解像度）を読み込み (PNG形式でRGBで読み込む)
    original_image_rgb = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)
    height, width = original_image_rgb.shape[:2]

    # 低解像度画像を生成（4倍のために元の1/4サイズにする）
    low_res_image_rgb = cv2.resize(original_image_rgb, (width // upscale_factor, height // upscale_factor), interpolation=cv2.INTER_LINEAR)

    # 低解像度画像を保存 (PNGで保存)
    low_res_image_path = Path(output_dir) / "low_resolution_image.png"
    cv2.imwrite(str(low_res_image_path), low_res_image_rgb)

    # ONNXモデルによる解像度アップ（AI超解像）
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # 低解像度画像の前処理: 0〜1に正規化し、適切な次元に変換
    input_image_prepared = low_res_image_rgb.astype(np.float32) / 255.0  # 0〜1に正規化
    input_image_prepared = input_image_prepared.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    input_image_prepared = np.expand_dims(input_image_prepared, axis=0).astype(np.float32)  # バッチサイズ追加

    # AI超解像による推論
    output_image_ai = session.run(None, {"input": input_image_prepared})[0]
    output_image_ai = np.squeeze(output_image_ai)  # バッチ次元を削除
    output_image_ai = output_image_ai.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    
    # 推論後の画像を0〜255の範囲に戻す
    output_image_ai = np.clip(output_image_ai * 255.0, 0, 255).astype(np.uint8)  # 0〜255に戻す

    # Bicubic補間による解像度アップ
    output_image_bicubic_rgb = cv2.resize(low_res_image_rgb, (width, height), interpolation=cv2.INTER_CUBIC)

    # 保存先ディレクトリを作成
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)

    # 保存 (PNGで保存)
    input_image_name = Path(input_image_path).stem  # 画像名を取得
    ai_upscaled_path = str(output_dir_path / f"{input_image_name}_ai_upscaled.png")
    bicubic_upscaled_path = str(output_dir_path / f"{input_image_name}_bicubic_upscaled.png")
    
    cv2.imwrite(ai_upscaled_path, output_image_ai)
    cv2.imwrite(bicubic_upscaled_path, output_image_bicubic_rgb)


if __name__ == "__main__":
    # 画像のパス、ONNXモデルのパス、保存先ディレクトリを指定
    input_image_path = "sample_generate/baboon.png"  # PNG画像を指定
    output_dir = "sample_generate/output_images"
    model_path = "submit/model.onnx"

    # 解像度を上げた画像を保存し、PSNRを計算（4倍スケール）
    upscale_image_and_calculate_psnr(input_image_path, output_dir, model_path)
