import tensorflow as tf

def trainmodel(model, xdata, ydata, args):
    NUMBER_EPOCHS = args["nepoch"]
    filepath = args["ckpt"]
    BATCH_SIZE = args["batchsize"]
    validation_split = args["valsplit"]

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        save_freq="epoch",
        # options=None,
    )
    hist = model.fit(
        x=xdata,
        y=ydata,
        epochs=NUMBER_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=validation_split,
        verbose=1,
        callbacks=[model_checkpoint_callback],
        class_weight={0: 1, 1: 5},
    )
    return hist

# tot_SU = 69159
# DS = 8502 -> 12.29%    (tot - DS )/DS -> 1 : 7  ----- BEST 1:2     is not strightfarward in the model, so better try and tuning, trial and error.
# DF = 4670 -> 6.75%     (tot - DF )/DF -> 1 : 14 ----- BEST 1:3
# ES = 2002 -> 2.89%     (tot - ES )/ES -> 1 : 34 ----- BEST 1:5
# EF = 1247 -> 1.80%     (tot - EF )/EF -> 1 : 54 ----- BEST 1:6
# RS1 = 221 -> 0.32%     (tot - RS )/RS -> 1 : 312 ----- BEST 1:22