import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Step 1: Load the image
image = cv2.imread("sample_images/airliners_02.jpg")

# Convert image to tensor format for the model
image_tensor = F.to_tensor(image)

# Step 2: Load the pre-trained object detection model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set model to evaluation mode (not training)

# Step 3: Make prediction (no gradients needed)
with torch.no_grad():
    output = model([image_tensor])[0]  # Output is a dictionary with boxes, labels, scores

# Step 4: Print results to understand what the model detected
print("Labels:", output['labels'])
print("Scores:", output['scores'])

# Step 5: Draw box if airplane is found
for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
    if label.item() == 5 and score.item() > 0.3:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Aircraft {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Step 6 (optional): Save the result image
cv2.imwrite("output_detected.jpg", image)

# Step 7: Show the image using matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis('off')
plt.title("Detected Aircraft")
plt.show()
