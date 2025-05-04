
class ConLearn:

    def train_and_evaluate(train_x, test_x, train_labels, test_labels):
        """
        create mode, start training, save in model folder, save performance in same folder"""
        
        print("Start training...")
        input_shape = train_x.shape[1]
        output_shape = train_labels.shape[1]  # Number of conflict columns
        
        print("train_and_evaluate::creating model...")
        model = ConLearn.create_model(input_shape, output_shape)
        print("train_and_evaluate:: Done creating model")

        print("train_and_evaluate::compiling model and train it...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss="binary_crossentropy",
            metrics=['accuracy']
        )
        
        history = model.fit(
            train_x, train_labels,
            epochs=12,
            batch_size=1024,
            validation_data=(test_x, test_labels),
            verbose=1
        )
        print("train_and_evaluate:: Done training model")
        
        # Save model
        model_id = str(uuid.uuid4())
        model_dir = f'Models/{model_id}'
        os.makedirs(model_dir, exist_ok=True)
        model.save(f'{model_dir}/model.keras')
        
        # Save training history plots
        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{model_dir}/losses.png')
        plt.close()
        
        plt.figure()
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{model_dir}/accuracy.png')
        plt.close()
        
        # Update model library
        model_library = pd.DataFrame({
            'ID': [model_id],
            **{col: [0] for col in range(input_shape)},
            'Conflict': [1]
        })
        model_library.to_csv('ConflictModelLibrary.csv', index=False)
        
        return model_id, history.history

 

