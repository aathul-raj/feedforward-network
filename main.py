import network
import mnist_loader

def main():
    # Load and prepare data
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    augmented_training_data = network.augment_data(training_data)

    # Define network architecture
    network_architecture = [784, 256, 128, 64, 10]

    # Set hyperparameters
    hyperparameters = {
        'epochs': 50,
        'mini_batch_size': 32,
        'learning_rate': 0.015,
        'lambda_reg': 0.01  # Add regularization parameter
    }

    # Initialize and train the network
    net = network.Network(network_architecture, lambda_reg=hyperparameters['lambda_reg'])
    net.SGD(augmented_training_data, 
            hyperparameters['epochs'], 
            hyperparameters['mini_batch_size'], 
            hyperparameters['learning_rate'], 
            test_data=test_data)

if __name__ == "__main__":
    main()