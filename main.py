import cv2
import numpy as np
import os 
import glob 
import csv 
import skimage.metrics
from skimage.metrics import mean_squared_error


# diretorios ----------------------------------------------------
RAW_DIR = 'raw-890'
REF_DIR = 'reference-890'
OUTPUT_DIR = 'output/'
CSV_FILENAME = 'metrics_results.csv'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#----------------------------------------------------------------


def dac_implementation(img):

    # Define tamanho max para o Mean filter
    hight, weight = img.shape[:2]
    max_size = max(hight, weight)  

    # matriz deve ser impar
    if max_size % 2 == 0:
        max_size +=1

    # OpenCV le na ordem BGR
    b_channel, g_channel, r_channel = cv2.split(img)


    # Utiliza CLAHE em cada channel para aumentar constraste da imagem 

    # Arigo omite os valores de limite e tamanho do bloco
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))   
    red_clahe = clahe.apply(r_channel)
    green_clahe = clahe.apply(g_channel)
    blue_clahe = clahe.apply(b_channel)


    # Blue_clahe = Base Layer Blue + Detail Layer Blue (Green is the same thing )

    # artigo tambem nao fala os falores do filtro bilateral
    r_bilateral = cv2.bilateralFilter(red_clahe, d=1, sigmaColor=1, sigmaSpace=1)
    g_bilateral = cv2.bilateralFilter(green_clahe, d=1, sigmaColor=1, sigmaSpace=1)
    b_bilateral = cv2.bilateralFilter(blue_clahe, d=1, sigmaColor=1, sigmaSpace=1)


    # converte para float32 para fazer a subtração e possuir valores negativos
    r_bilateral = r_bilateral.astype(np.float32)
    g_bilateral = g_bilateral.astype(np.float32)
    b_bilateral = b_bilateral.astype(np.float32)

    # Separar a Base Layer da Detail Layer de cada canal (R nao vai ter detail layer)
    # mean filter é o cv2.blur
    r_baseLayer = cv2.blur(r_bilateral, (max_size, max_size))
    g_baseLayer = cv2.blur(g_bilateral, (max_size, max_size))
    b_baseLayer = cv2.blur(b_bilateral, (max_size, max_size))


    # Detail Layer do Green e Blue
    g_detailLayer = g_bilateral - g_baseLayer
    b_detailLayer = b_bilateral - b_baseLayer

    # meadia para conferir qual canal tem mais predominancia na imagem     
    g_average = np.mean(g_bilateral)
    b_average = np.mean(b_bilateral)

    # se a imagem conter mais azul, use o detail layer do azul, se verde for predominante use o detail layer do verde
    # r_detailLayer é o Rcp (R compensation)
    if(g_average > b_average):
        # verde predominante
        Rcp = r_baseLayer + g_detailLayer

    else:
        # azul predominante
        Rcp = r_baseLayer + b_detailLayer


    # Rcp contém todo o R Channel

    # Utiliza Gray World Algorithm para corrigir as cores


    # media do R channel
    r_average = np.mean(Rcp)

    # FORMULA GRAY
    gray = (r_average + g_average + b_average) / 3

    # K é o scale parameter 
    # FORMULA Kr
    k_r = 0.8 * (gray / r_average)   # 0.8 é o valor alfa dado no artigo

    # FORMULA Kc  
    k_g = gray / g_average
    k_b = gray / b_average


    # Multiplicações Gray World
    r_gw = k_r * Rcp 
    g_gw = k_g * g_bilateral 
    b_gw = k_b * b_bilateral

    # FORMULA IRout
    r_out = np.clip(r_gw, 0, 255).astype(np.uint8)

    # FORMULA ICout
    g_out = np.clip(g_gw, 0, 255).astype(np.uint8)
    b_out = np.clip(b_gw, 0, 255).astype(np.uint8)

    # Juntar tudo (BGR)
    final_image = cv2.merge((b_out, g_out, r_out))

    return final_image




# pega todas as imagens da pasta raw
image_paths = glob.glob(os.path.join(RAW_DIR, '*.*'))  # aceita png, jpg, etc

# lista de resultados
results    = []
total_mse  = []
total_psnr = []
total_ssim = []


print(f"Starting the processing of {len(image_paths)} images =>")

for raw_path in image_paths:
    filename = os.path.basename(raw_path)
    reference_path = os.path.join(REF_DIR, filename)


    # verifica se a imagem de referencia existe
    if not os.path.exists(reference_path):
        print(f"Error: {filename} has no reference image! (Skipping)")
        continue

    # carrega as imagens
    img_raw = cv2.imread(raw_path)
    img_ref = cv2.imread(reference_path)

    # processa a imagem
    processed_image = dac_implementation(img_raw)

    # salva
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), processed_image)


    # calcula as metricas
    # channel_axis=-1 parametro do scikit-image para avisar que a imagem tem cor
    mse_value = mean_squared_error(img_ref, processed_image)
    psnr_value = skimage.metrics.peak_signal_noise_ratio(img_ref, processed_image)
    ssim_value = skimage.metrics.structural_similarity(img_ref, processed_image, channel_axis=-1)


    # guarda na memoria
    results.append([filename, mse_value, psnr_value, ssim_value])
    total_mse.append(mse_value)
    total_psnr.append(psnr_value)
    total_ssim.append(ssim_value)

    print(f"{filename}: SUCCESS! | MSE: {mse_value:.2f} | PSNR: {psnr_value:.2f} | SSIM: {ssim_value:.4f}")


    # Relatorio Final ---------------------------------------------------------------------------
if results:

    # Salvar em CSV
    with open(CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File', 'MSE', 'PSNR', 'SSIM'])
        writer.writerows(results)
    
    # Calcular médias
    mean_mse = np.mean(total_mse)
    mean_psnr = np.mean(total_psnr)
    mean_ssim = np.mean(total_ssim)
    
    print("\n" + "="*40)
    print("🎯 PROCESSING COMPLETE!")
    print("="*40)
    print(f"Total image processed: {len(results)}")
    print(f"FINAL MEAN MSE: {mean_mse:.4f}")
    print(f"FINAL MEAN PSNR: {mean_psnr:.4f}")
    print(f"FINAL MEAN SSIM: {mean_ssim:.4f}")
    print(f"Detailed Results Saved On: {CSV_FILENAME}")
else:
    print("\nNo image was processed. Revise the file paths...")