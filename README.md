# CNN для розпізнавання рукописних цифр MNIST

Лабораторна робота 5: згорткова нейронна мережа (Keras / TensorFlow), навчена на MNIST.

## Модель

Файл: `model/cnn_mnist_final.keras`

- Архітектура: 5 згорткових шарів (32, 32, 64, 64, 128) з BatchNormalization, 3 шари MaxPooling, Dropout 0,25–0,4, Dense 256 → 64 → 10 (softmax).
- Оптимізатор Adam, функція втрат categorical_crossentropy.
- Навчання: `validation_split=0.1`, EarlyStopping за `val_loss` (patience 3, restore_best_weights).
- **Точність на тестовій вибірці MNIST: 99,39% (помилка 0,61%).**
- Розпізнавання 10 зображень датасету: 10 з 10; власних цифр: 8 з 10.

## Структура

```
model/       збережена модель
notebooks/   Colab-ноутбуки: навчання (task4_part1) і розпізнавання (task4_part2)
digits/      власні зображення 0.png ... 9.png
predict.py   розпізнавання з командного рядка
```

## Запуск

```bash
pip install -r requirements.txt
python predict.py digits/0.png digits/1.png
```

Зображення: темна цифра на світлому фоні; скрипт сам вирівнює фон, обрізає цифру, масштабує до 28×28 та інвертує.
