import cv2
import numpy as np
import os 
import glob 
import csv 
import skimage.metrics
from skimage.metrics import mean_squared_error


# diretorios ----------------------------------------------------
RAW_DIR = 'UIEB_Raw'
REF_DIR = 'UIEB_Reference'
OUTPUT_DIR = 'output/'
CSV_FILENAME = 'metrics_results.csv'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#----------------------------------------------------------------

# recebe a matriz de pixels diretamente

def dac_implementation(img):

    # Define tamanho max para o mean filter
    hight, weight = img.shape[:2]
    max_size = max(hight, weight)  

    # opencv exige que a matriz seja impar
    if max_size % 2 == 0:
        max_size +=1

    # OpenCV le na ordem BGR
    b_channel, g_channel, r_channel = cv2.split(img)

    # Utiliza CLAHE em cada channel para aumentar constraste da imagem
    # Arigo omite os valores de limite e tamanho do bloco
    clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(8,8))   # testar os valores
    red_enhanced = clahe.apply(r_channel)
    green_enhanced = clahe.apply(g_channel)
    blue_enhanced = clahe.apply(b_channel)

    # converte para float para manter os valores negativos
    # artigo tambem nao fala os falores do filtro bilateral
    r_enhanced = cv2.bilateralFilter(red_enhanced.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)
    g_enhanced = cv2.bilateralFilter(green_enhanced.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)
    b_enhanced = cv2.bilateralFilter(blue_enhanced.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)


    # Separar a Base Layer da Detail Layer de cada canal (R nao vai ter detail layer)
    # mean filter é o cv2.blur
    r_baseLayer = cv2.blur(r_enhanced, (max_size, max_size))
    g_baseLayer = cv2.blur(g_enhanced, (max_size, max_size))
    b_baseLayer = cv2.blur(b_enhanced, (max_size, max_size))

    g_detailLayer = g_enhanced - g_baseLayer
    b_detailLayer = b_enhanced - b_baseLayer

    # Criar Detail Layer do canal R a partir da Base Layer R e Detail Layer do G e B  (Criando o R channel completo)
    g_average = np.mean(g_enhanced)
    b_average = np.mean(b_enhanced)

    # se a imagem conter mais azul, use o detail layer do azul, se verde for predominante use o detail layer do verde
    # r_detailLayer é o r_cp (compensation)
    if(g_average > b_average):
        # verde predominante
        r_detailLayer = cv2.add(r_baseLayer, g_detailLayer)

    else:
        # azul predominante
        r_detailLayer = cv2.add(r_baseLayer, b_detailLayer)


    # elimina valores negativos antes do gray world
    r_detailLayer = np.clip(r_detailLayer, 0.0, 255.0)


    # Utiliza Gray World Algorithm para corrigir as cores
    r_average = np.mean(r_detailLayer)
    gray = (r_average + g_average + b_average) / 3

    # K é o scale parameter 
    k_r = 0.8 * (gray / r_average)   # 0.8 é o valor alfa dado no artigo
    k_g = gray / g_average
    k_b = gray / b_average

    #DEBUG
    #print(f"g_avg={g_average:.2f} b_avg={b_average:.2f} r_avg={r_average:.2f} gray={gray:.2f}")
    #print(f"k_r={k_r:.2f} k_g={k_g:.2f} k_b={k_b:.2f}")


    r_output = np.clip(k_r * r_detailLayer, 0, 255).astype(np.uint8)
    # saida do G e B a formula utiliza enhanced
    g_output = np.clip(k_g * g_enhanced, 0, 255).astype(np.uint8)
    b_output = np.clip(k_b * b_enhanced, 0, 255).astype(np.uint8)

    # Juntar tudo (BGR)
    #final_image = cv2.merge((b_output, g_output, r_output))
    final_image = cv2.merge((b_output, g_output, r_output))

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