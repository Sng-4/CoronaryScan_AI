from locust import HttpUser, task, between, events
import os

# Define the path to a dummy image for testing
# You must have a file with this name in the folder, or the test will skip predictions
TEST_IMAGE_PATH = "test_xray.png"

class ArcadeUser(HttpUser):
    # Simulate a user waiting between 1 and 3 seconds between actions
    wait_time = between(1, 3)
    
    def on_start(self):
        """
        Runs once when a simulated user is hatched.
        We load the image into memory here to avoid disk I/O bottlenecks during the test.
        """
        self.image_data = None
        if os.path.exists(TEST_IMAGE_PATH):
            with open(TEST_IMAGE_PATH, "rb") as f:
                self.image_data = f.read()
        else:
            print(f"⚠️ WARNING: {TEST_IMAGE_PATH} not found. Prediction tests will fail.")

    @task(1)
    def health_check(self):
        """
        Lightweight task: Checks if the API is responsive.
        """
        self.client.get("/")

    @task(3)
    def predict_stenosis(self):
        """
        Heavy task: Simulates uploading an X-ray for diagnosis.
        This tests the CPU/Memory limits of your Docker container.
        """
        if self.image_data:
            # We use a tuple (filename, file_bytes, mime_type) for the files argument
            files = {
                "file": ("test_xray.png", self.image_data, "image/png")
            }
            self.client.post("/predict", files=files)