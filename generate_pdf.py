from fpdf import FPDF
from fpdf.enums import XPos, YPos
import textwrap

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Vision Intelligence CV Engineer - Live Task Submission', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(5)

pdf = PDF()
pdf.add_page()
pdf.set_font("helvetica", size=11)

q1 = "1. How would you scale this from one live camera to 500 cameras streaming at once? Where would it break first?"
a1 = "To scale to 500 cameras, the monolithic script must be decoupled into a distributed microservices architecture using message brokers (e.g., Kafka) and scalable inference servers (e.g., NVIDIA Triton). The system would break first at the hardware inference bottleneck; running 500 parallel YOLO-Pose streams on a single node will exhaust VRAM and compute immediately. Resolving this requires lowering the processing frame rate (e.g., analyzing 3-5 FPS instead of 30), batching frames from multiple streams, and horizontally scaling across a GPU cluster."

q2 = "2. How would you avoid double-counting or losing track of a person if they briefly leave the camera's view?"
a2 = "First, the tracking algorithm's memory buffer (e.g., track_buffer in BoT-SORT) must be extended to keep the track identity \"alive\" for several seconds during occlusions. For longer disappearances where the tracker drops the ID, a lightweight Re-Identification (Re-ID) mechanism is required. By extracting and storing an appearance embedding (like an HSV color histogram or a fast deep embedding) when a track exits, we can compare newly appearing tracks against recent exits and merge them if the similarity is high, preventing double-counting."

q3 = "3. How would you handle a camera feed that's consistently blurry or poor quality - flag it, skip it, or something else?"
a3 = "Consistently poor feeds should not be blindly processed or silently skipped, as both pollute the database and hide hardware failures. The pipeline should calculate a rolling baseline of frame sharpness (using Laplacian variance) and automatically trigger a \"Degraded Camera\" alert to the IT/maintenance team if the median drops below an acceptable threshold. During this degraded state, the system should continue processing but explicitly tag all generated analytics in the database with a \"Low Confidence\" flag so downstream consumers know the data is unreliable until the lens is cleaned."

for q, a in [(q1, a1), (q2, a2), (q3, a3)]:
    pdf.set_font("helvetica", 'B', 12)
    pdf.multi_cell(0, 7, q)
    pdf.ln(2)
    pdf.set_font("helvetica", '', 11)
    pdf.multi_cell(0, 6, a)
    pdf.ln(8)

pdf.output("submission_answers.pdf")
print("PDF generated successfully.")
