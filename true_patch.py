import os
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class PatchMeta:
    x: int  # левая координата патча в исходном изображении
    y: int  # верхняя координата патча в исходном изображении
    filename: str  # имя исходного файла


def extract_overlapping_patches(
    image: Image.Image,
    patch_size: int = 1024,
    overlap: int = 256,
) -> Tuple[List[Image.Image], List[PatchMeta], Tuple[int, int]]:
    """
    Извлекает патчи из изображения с заданным размером и перекрытием
    """
    img = image.convert("RGB")
    W, H = img.size
    
    # Убедимся, что перекрытие не превышает размер патча
    overlap = min(overlap, patch_size - 1)
    step = patch_size - overlap

    patches: List[Image.Image] = []
    metas: List[PatchMeta] = []

    # Вычисляем координаты для нарезки с перекрытием
    xs = []
    x = 0
    while x + patch_size <= W:
        xs.append(x)
        x += step
    # Добавляем последний патч, если не доходим до края
    if xs and xs[-1] + patch_size < W:
        xs.append(W - patch_size)
    elif not xs and W >= patch_size:
        xs.append(W - patch_size)
    
    ys = []
    y = 0
    while y + patch_size <= H:
        ys.append(y)
        y += step
    if ys and ys[-1] + patch_size < H:
        ys.append(H - patch_size)
    elif not ys and H >= patch_size:
        ys.append(H - patch_size)

    for y in ys:
        for x in xs:
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
            metas.append(PatchMeta(x=x, y=y, filename=""))

    return patches, metas, (W, H)


def downsample_patch(patch: Image.Image, target_size: int) -> Image.Image:
    """
    Уменьшает патч в target_size раз с использованием LANCZOS ресемплинга
    """
    if target_size >= patch.size[0]:
        return patch.copy()
    
    # Вычисляем коэффициент уменьшения
    scale_factor = target_size / patch.size[0]
    
    # Уменьшаем изображение
    downsampled = patch.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return downsampled


def process_images(
    input_dir: str,
    output_base_dir: str,
    patch_size: int = 1024,
    overlap: int = 256
) -> None:
    """
    Обрабатывает все изображения в папке:
    - нарезает патчи максимального размера (1024x1024) с перекрытием
    - создаёт уменьшенные версии этих патчей для всех размеров
    - сохраняет в соответствующие папки
    """
    input_path = Path(input_dir)
    output_path = Path(output_base_dir)
    
    # Создаём базовую директорию для выходных данных
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Определяем размеры для уменьшения (от 1024 до 32 с шагом деления на 2)
    target_sizes = []
    current_size = patch_size
    while current_size >= 32:
        target_sizes.append(current_size)
        current_size //= 2
    
    print(f"Будут созданы патчи размеров: {target_sizes}")
    print(f"Параметры нарезки: размер={patch_size}x{patch_size}, перекрытие={overlap}")
    
    # Создаём папки для каждого размера
    size_folders = {}
    for size in target_sizes:
        folder_name = f"patches_{size}x{size}"
        size_folder = output_path / folder_name
        size_folder.mkdir(parents=True, exist_ok=True)
        size_folders[size] = size_folder
    
    # Получаем все изображения в папке
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"\nНайдено изображений: {len(image_files)}")
    
    total_patches = 0
    patches_by_size = {size: 0 for size in target_sizes}
    
    for img_file in image_files:
        print(f"\nОбработка: {img_file.name}")
        
        try:
            # Загружаем изображение
            img = Image.open(img_file)
            print(f"  Размер изображения: {img.size}")
            
            # Извлекаем патчи максимального размера с перекрытием
            high_res_patches, metas, orig_size = extract_overlapping_patches(
                img,
                patch_size=patch_size,
                overlap=overlap
            )
            
            print(f"  Создано HR патчей: {len(high_res_patches)}")
            total_patches += len(high_res_patches)
            
            # Обрабатываем каждый HR патч
            for idx, hr_patch in enumerate(high_res_patches):
                # Сохраняем HR патч (оригинальный размер)
                hr_filename = f"{img_file.stem}_patch_{idx:04d}.png"
                hr_save_path = size_folders[patch_size] / hr_filename
                hr_patch.save(hr_save_path)
                patches_by_size[patch_size] += 1
                
                # Создаём уменьшенные версии для всех остальных размеров
                for target_size in target_sizes:
                    if target_size == patch_size:
                        continue  # HR патч уже сохранён
                    
                    # Уменьшаем патч до целевого размера
                    downsampled_patch = downsample_patch(hr_patch, target_size)
                    
                    # Сохраняем уменьшенный патч
                    lr_filename = f"{img_file.stem}_patch_{idx:04d}.png"
                    lr_save_path = size_folders[target_size] / lr_filename
                    downsampled_patch.save(lr_save_path)
                    patches_by_size[target_size] += 1
            
            print(f"  Все патчи успешно сохранены")
            
        except Exception as e:
            print(f"  Ошибка при обработке {img_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*50}")
    print(f"Обработка завершена!")
    print(f"Всего создано патчей: {total_patches * len(target_sizes)}")
    print(f"\nСтатистика по размерам:")
    for size in target_sizes:
        print(f"  {size}x{size}: {patches_by_size[size]} патчей")
    print(f"\nПапки с патчами сохранены в: {output_path}")
    
    # Выводим информацию о содержимом папок
    for size, folder in size_folders.items():
        num_files = len(list(folder.glob("*.png")))
        print(f"  {folder.name}: {num_files} патчей")


def verify_patch_consistency(
    output_base_dir: str,
    patch_size: int = 1024
) -> None:
    """
    Проверяет согласованность патчей между разными размерами
    """
    output_path = Path(output_base_dir)
    
    # Определяем размеры для проверки
    target_sizes = []
    current_size = patch_size
    while current_size >= 32:
        target_sizes.append(current_size)
        current_size //= 2
    
    # Проверяем первую папку с патчами
    hr_folder = output_path / f"patches_{patch_size}x{patch_size}"
    if not hr_folder.exists():
        print("Папка с HR патчами не найдена")
        return
    
    # Берем первый патч для проверки
    hr_patches = list(hr_folder.glob("*.png"))
    if not hr_patches:
        print("Патчи не найдены")
        return
    
    print("\nПроверка согласованности патчей:")
    for hr_patch_path in hr_patches[:3]:  # Проверяем первые 3 патча
        print(f"\nПроверка: {hr_patch_path.name}")
        
        # Загружаем HR патч
        hr_patch = Image.open(hr_patch_path)
        print(f"  HR патч: {hr_patch.size}")
        
        # Проверяем LR патчи
        for size in target_sizes:
            if size == patch_size:
                continue
                
            lr_folder = output_path / f"patches_{size}x{size}"
            lr_patch_path = lr_folder / hr_patch_path.name
            
            if lr_patch_path.exists():
                lr_patch = Image.open(lr_patch_path)
                print(f"  {size}x{size} патч: {lr_patch.size}")
                
                # Проверяем соотношение размеров
                expected_size = size
                if lr_patch.size[0] != expected_size:
                    print(f"    ВНИМАНИЕ: Несоответствие размера! Ожидалось {expected_size}, получено {lr_patch.size[0]}")
            else:
                print(f"  {size}x{size} патч: НЕ НАЙДЕН")


if __name__ == "__main__":
    # Пути к папкам
    input_directory = "/home/sigov.s@agtu.ru/Рабочий стол/data/imgs"
    output_directory = "/home/sigov.s@agtu.ru/Рабочий стол/data/"
    
    # Параметры нарезки патчей
    PATCH_SIZE = 1024      # Начальный размер патча
    INITIAL_OVERLAP = 256  # Перекрытие патчей
    
    # Запускаем обработку
    process_images(
        input_dir=input_directory,
        output_base_dir=output_directory,
        patch_size=PATCH_SIZE,
        overlap=INITIAL_OVERLAP
    )
    
    # Проверяем согласованность созданных патчей
    verify_patch_consistency(output_directory, PATCH_SIZE)