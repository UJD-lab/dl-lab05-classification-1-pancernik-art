import tensorflow as tf
from tensorflow.keras import layers, applications
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pathlib
import gradio as gr
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns

IMG_SIZE = 224
BATCH_SIZE = 32

model = None
train_ds = None
val_ds = None
class_names = None
NUM_CLASSES = None

def load_data():
    global train_ds, val_ds, class_names, NUM_CLASSES

    url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=url, untar=True)
    data_dir = pathlib.Path(data_dir)
    if (data_dir / 'flower_photos').exists():
        data_dir = data_dir / 'flower_photos'

    full_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=True,
        seed=42
    )

    class_names = full_ds.class_names
    NUM_CLASSES = len(class_names)

    total = sum(1 for _ in full_ds)
    train_size = int(0.8 * total)

    train_ds = full_ds.take(train_size)
    val_ds   = full_ds.skip(train_size)

    def normalize(images, labels):
        return tf.cast(images, tf.float32) / 255.0, labels

    def augment(images, labels):
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_brightness(images, 0.2)
        images = tf.image.random_contrast(images, 0.8, 1.2)
        return images, labels

    train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE) \
                       .map(augment,    num_parallel_calls=tf.data.AUTOTUNE) \
                       .prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE) \
                   .prefetch(tf.data.AUTOTUNE)

def build_model():
    global model, NUM_CLASSES

    base_model = applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(NUM_CLASSES)(x)

    model = tf.keras.Model(inputs, outputs)
    return base_model

def make_plots(acc, val_acc, loss, val_loss, phase1_epochs, y_true=None, y_pred=None, metrics_text=None):
    plt.close('all')
    fig = plt.figure(figsize=(11, 7), dpi=90)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    epochs = list(range(1, len(acc) + 1))

    ax1.plot(epochs, acc, label='accuracy', color='royalblue')
    ax1.plot(epochs, val_acc, label='val_accuracy', color='tomato')
    if len(acc) > phase1_epochs:
        ax1.axvline(x=phase1_epochs + 0.5, color='gray', linestyle='--', label='fine-tuning start')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0.0, 1.0])
    ax1.set_title('Accuracy')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, loss, label='loss', color='royalblue')
    ax2.plot(epochs, val_loss, label='val_loss', color='tomato')
    if len(loss) > phase1_epochs:
        ax2.axvline(x=phase1_epochs + 0.5, color='gray', linestyle='--', label='fine-tuning start')
    ax2.set_xlabel('Epoka')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    if y_true is not None and y_pred is not None:
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax3, annot_kws={'size': 8})
        ax3.set_xlabel('Przewidziana klasa')
        ax3.set_ylabel('Prawdziwa klasa')
        ax3.tick_params(axis='x', rotation=30, labelsize=7)
        ax3.tick_params(axis='y', rotation=0, labelsize=7)
    else:
        ax3.text(0.5, 0.5, 'Dostępna po\nzakończeniu treningu',
                 ha='center', va='center', fontsize=11, color='gray',
                 transform=ax3.transAxes)
        ax3.set_xticks([])
        ax3.set_yticks([])
    ax3.set_title('Confusion Matrix')

    ax4.axis('off')
    if metrics_text:
        ax4.text(0.1, 0.55, metrics_text, transform=ax4.transAxes,
                 fontsize=12, verticalalignment='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'Metryki dostępne\npo zakończeniu treningu',
                 ha='center', va='center', fontsize=11, color='gray',
                 transform=ax4.transAxes)
    ax4.set_title('Precision / Recall / F1')

    plt.tight_layout()
    return fig

def evaluate_metrics():
    all_true, all_pred = [], []
    for images, labels in val_ds:
        logits = model(images, training=False)
        preds = tf.argmax(logits, axis=1).numpy()
        all_true.extend(labels.numpy())
        all_pred.extend(preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average='macro', zero_division=0
    )
    metrics_text = (
        f"Precision (macro): {precision:.3f}\n"
        f"Recall    (macro): {recall:.3f}\n"
        f"F1-score  (macro): {f1:.3f}"
    )
    return metrics_text, all_true, all_pred

def train(epochs_phase1, epochs_phase2):
    global model, train_ds, val_ds, class_names, NUM_CLASSES

    yield None, "Wczytywanie danych..."
    if train_ds is None:
        load_data()

    yield None, "Budowanie modelu..."
    model = None
    base_model = build_model()

    acc, val_acc, loss, val_loss = [], [], [], []
    epochs_phase1 = int(epochs_phase1)
    epochs_phase2 = int(epochs_phase2)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    for epoch in range(epochs_phase1):
        hist = model.fit(
            train_ds, epochs=epoch + 1, initial_epoch=epoch,
            validation_data=val_ds, verbose=0
        )
        acc.append(hist.history['accuracy'][0])
        val_acc.append(hist.history['val_accuracy'][0])
        loss.append(hist.history['loss'][0])
        val_loss.append(hist.history['val_loss'][0])
        fig = make_plots(acc, val_acc, loss, val_loss, epochs_phase1)
        status = f"Faza 1 — epoka {epoch + 1}/{epochs_phase1} | acc: {acc[-1]:.3f} | val_acc: {val_acc[-1]:.3f} | loss: {loss[-1]:.3f}"
        yield fig, status

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    for epoch in range(epochs_phase2):
        hist = model.fit(
            train_ds, epochs=epoch + 1, initial_epoch=epoch,
            validation_data=val_ds, verbose=0
        )
        acc.append(hist.history['accuracy'][0])
        val_acc.append(hist.history['val_accuracy'][0])
        loss.append(hist.history['loss'][0])
        val_loss.append(hist.history['val_loss'][0])
        fig = make_plots(acc, val_acc, loss, val_loss, epochs_phase1)
        status = f"Faza 2 (fine-tuning) — epoka {epoch + 1}/{epochs_phase2} | acc: {acc[-1]:.3f} | val_acc: {val_acc[-1]:.3f} | loss: {loss[-1]:.3f}"
        yield fig, status

    metrics_text, all_true, all_pred = evaluate_metrics()
    fig = make_plots(acc, val_acc, loss, val_loss, epochs_phase1, all_true, all_pred, metrics_text)
    fig.savefig("./images/wykres_treningu.png", dpi=150, bbox_inches='tight')
    yield fig, "Trening zakończony!"

def predict(image):
    if model is None:
        return "Model nie jest jeszcze wytrenowany. Najpierw uruchom trening."

    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, 0)

    logits = model(image, training=False)
    probs = tf.nn.softmax(logits[0]).numpy()
    top5_idx = np.argsort(probs)[::-1][:5]

    result = ""
    for i, idx in enumerate(top5_idx):
        bar = "█" * int(probs[idx] * 30)
        result += f"{i+1}. {class_names[idx]:<30} {probs[idx]*100:5.1f}%  {bar}\n"

    return result

with gr.Blocks(title="Flowers - Transfer Learning") as demo:
    gr.Markdown("# Flowers — Transfer Learning z MobileNetV2")

    with gr.Tab("Trening"):
        with gr.Row():
            epochs1 = gr.Slider(1, 20, value=5, step=1, label="Epoki faza 1 (zamrożona baza)")
            epochs2 = gr.Slider(1, 20, value=5, step=1, label="Epoki faza 2 (fine-tuning)")
        btn_train = gr.Button("Start trening", variant="primary")
        status_box = gr.Textbox(label="Status", interactive=False)
        plot = gr.Plot(label="Wyniki treningu")

        def save_plot():
            import os
            path = "./images/wykres_treningu.png"
            if os.path.exists(path):
                return path
            return None

        btn_save = gr.Button("Pobierz wykres (PNG)", variant="secondary")
        download_file = gr.File(label="Plik do pobrania", visible=False)

        btn_train.click(
            fn=train,
            inputs=[epochs1, epochs2],
            outputs=[plot, status_box]
        )

        btn_save.click(
            fn=save_plot,
            inputs=[],
            outputs=[download_file]
        )

    with gr.Tab("Klasyfikacja obrazu"):
        gr.Markdown("Wgraj zdjęcie kwiatu, model przewidzi klasę.")
        image_input = gr.Image(label="Zdjęcie", type="numpy")
        btn_predict = gr.Button("Przewidywanie klasy", variant="primary")
        prediction_output = gr.Textbox(label="Top 5 predykcji", lines=6)

        btn_predict.click(
            fn=predict,
            inputs=image_input,
            outputs=prediction_output
        )

demo.launch()