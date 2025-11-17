import cv2
import numpy as np
import os
# Предполагаем, что эти модули вы создали по моему плану
from face_detector import FaceDetector 
from overlay_transformer import OverlayTransformer 

# --- 1. Константы и Пути к Файлам ---
# Убедитесь, что этот файл находится в 'resources/models/'
MODEL_PATH = "resources/models/shape_predictor_68_face_landmarks.dat"
# Убедитесь, что это изображение PNG с прозрачным фоном находится в 'resources/accessories/'
ACCESSORY_PATH = "resources/accessories/glasses.png"

def main():
    """Основной цикл приложения Real-Time Virtual Try-On."""
    
    # Проверка наличия файла модели
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Файл модели Dlib не найден по пути: {MODEL_PATH}")
        print("Пожалуйста, скачайте 'shape_predictor_68_face_landmarks.dat' и поместите его туда.")
        return
        
    # Проверка наличия файла аксессуара
    if not os.path.exists(ACCESSORY_PATH):
        print(f"Ошибка: Файл аксессуара не найден по пути: {ACCESSORY_PATH}")
        print("Пожалуйста, создайте или скопируйте PNG-файл очков.")
        return

    try:
        # Инициализация FaceDetector (Шаг 2)
        face_detector = FaceDetector(MODEL_PATH)
        # Инициализация OverlayTransformer (Шаг 3)
        overlay_transformer = OverlayTransformer(ACCESSORY_PATH)
    except Exception as e:
        print(f"Критическая ошибка инициализации: {e}")
        return

    # Инициализация захвата видео (0 - обычно встроенная веб-камера)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру.")
        return

    print("Камера успешно открыта. Нажмите 'q' для выхода.")

    while True:
        # 1. Захват кадра
        ret, frame = cap.read()
        
        if not ret:
            print("Ошибка: Не удалось получить кадр.")
            break
            
        # 2. Детекция Ландмарков
        landmarks = face_detector.detect_landmarks(frame)
        
        if landmarks is not None:
            # Опционально: Рисуем ландмарки, чтобы убедиться, что они найдены
            # (Этот код можно удалить после успешной отладки)
            # for (x, y) in landmarks:
            #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            # 3. Геометрическая Трансформация и Наложение
            try:
                # В этой функции происходит расчет матрицы, трансформация и блендинг
                frame = overlay_transformer.overlay_accessory(frame, landmarks)
            except Exception as e:
                # Отладочный вывод, если трансформация не удалась (например, неверные ландмарки)
                print(f"Ошибка наложения: {e}") 
        
        # Отображение кадра
        cv2.imshow('Virtual Try-On (Press q to exit)', frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()