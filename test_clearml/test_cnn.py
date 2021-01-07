import os
from tempfile import gettempdir
import tensorflow as tf
from clearml import Task


def main():
    task = Task.init(project_name="pbl_example", task_name="tf2 mnist args")
    parameters = {'epochs': 5}
    task.connect(parameters)

    ds = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = ds.load_data()
    x_test, x_train = x_test.reshape((10000, 28, 28, 1)), x_train.reshape(60000, 28, 28, 1)
    x_test, x_train = x_test/255.0, x_train/255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(20, (5, 5), padding="SAME", activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2, padding="SAME"),
        tf.keras.layers.Conv2D(40, (5, 5), padding="SAME", activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2, padding="SAME"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=parameters['epochs'])
    optimizer = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(gettempdir(), 'tf_ckpts'), max_to_keep=3)
    save_path = manager.save()
    test_loss, test_acc = model.evaluate(x_test, y_test)
    # model.save_weights('./model')
    print(save_path)


if __name__ == '__main__':
    main()
