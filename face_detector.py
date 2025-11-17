# face_detector.py

import dlib
import cv2
import numpy as np
from imutils import face_utils

class FaceDetector:
    """
    Класс для обнаружения лица и извлечения 68 ключевых точек Dlib.
    """
    def __init__(self, model_path):
        """
        Инициализирует детектор лица Dlib и предиктор формы.
        :param model_path: Путь к файлу shape_predictor_68_face_landmarks.dat.
        """
        try:
            # Dlib's HOG-based face detector
            self.detector = dlib.get_frontal_face_detector()
            # Dlib's 68-point shape predictor
            self.predictor = dlib.shape_predictor(model_path)
            print("Dlib FaceDetector инициализирован.")
        except Exception as e:
            print(f"Ошибка инициализации Dlib: {e}")
            raise

    def detect_landmarks(self, frame):
        """
        Находит лицо и возвращает массив (68, 2) координат ландмарков.
        
        :param frame: Кадр OpenCV (BGR).
        :return: Массив ландмарков (68, 2) или None.
        """
        # Dlib лучше работает на оттенках серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Находим все лица на кадре
        rects = self.detector(gray, 0)
        
        if len(rects) == 0:
            return None # Лицо не найдено

        # Берем только первое найденное лицо для простоты
        rect = rects[0]
        
        # Применяем предиктор для получения 68 ландмарков
        shape = self.predictor(gray, rect)
        
        # Преобразуем объект dlib shape в массив NumPy (68, 2)
        landmarks = face_utils.shape_to_np(shape)
        
        return landmarks

# Внимание: для работы этого модуля необходимо установить imutils: pip install imutils
# или реализовать преобразование dlib.shape в numpy самостоятельно.