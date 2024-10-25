from utils import * 
from dotenv import load_dotenv

# Main function
def main(image_path, n_clusters=4):
    # Load and process image
    original_img = load_image(image_path)
    img_data = reshape_image(original_img)

    # Build and train the autoencoder
    autoencoder = build_autoencoder()
    train_autoencoder(autoencoder, img_data.astype('float32') / 255.0, epochs=30)

    # Use the autoencoder to encode the data
    encoded_data = autoencoder.predict(img_data.astype('float32') / 255.0)

    # Plot original image distribution
    plot_3d_scatter(img_data, img_data, "Original Image Color Distribution")

    # Perform k-means on encoded data
    kmeans = perform_kmeans(encoded_data, n_clusters)

    # Plot clustered image distribution
    plot_3d_scatter_res(encoded_data, kmeans.labels_, "Clustered Encoded Image Distribution")

    # Reconstruct segmented image
    segmented_img = reconstruct_image(kmeans.labels_, kmeans.cluster_centers_, original_img.shape)

    # Display original and segmented images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(original_img)
    ax1.set_title("Original Image")
    ax1.axis('off')
    ax2.imshow(segmented_img.astype(np.uint8))
    ax2.set_title("Segmented Image")
    ax2.axis('off')
    plt.show()

# Run the main function
if __name__ == "__main__":
    image_path = "assets/lena_color.png"  # Replace with your image path
    main(image_path)
