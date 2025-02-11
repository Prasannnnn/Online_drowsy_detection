from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import matplotlib.pyplot as plt
import os

def create_pdf_report(events, start_time, stop_time, report_path):
    timestamps = [event['timestamp'] for event in events]
    statuses = [event['status'] for event in events]
    names = [event['name'] for event in events]

    # Plot graph
    plt.figure(figsize=(10, 5))
    plt.bar(timestamps, [1]*len(timestamps), color='blue', label='Status')
    plt.title('Drowsiness Events')
    plt.xticks(rotation=45)
    plt.tight_layout()
    graph_path = "static/images/graph.png"
    plt.savefig(graph_path)
    plt.close()

    # Generate PDF
    c = canvas.Canvas(report_path, pagesize=letter)
    c.drawString(100, 750, "Drowsiness Detection Report")
    c.drawString(100, 735, f"Start Time: {start_time}")
    c.drawString(100, 720, f"Stop Time: {stop_time}")

    c.drawImage(graph_path, 100, 500, width=400, height=200)

    y = 480
    for event in events:
        c.drawString(100, y, f"{event['timestamp']}: {event['name']} - {event['status']}")
        y -= 15
        if y < 50:
            c.showPage()
            y = 750

    c.save()
    os.remove(graph_path)
