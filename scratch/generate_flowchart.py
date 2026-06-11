import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_flowchart():
    # Setup premium styling
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif'
    })
    
    # 14.5 inches wide, 3.2 inches high - wide aspect ratio to prevent clipping
    fig, ax = plt.subplots(figsize=(14.5, 3.2), dpi=300)
    ax.axis('off')
    
    # Define boxes helper
    def draw_box(x, y, w, h, text, title, bg_color, border_color, text_color='#1e293b'):
        # Rounded box
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03",
            linewidth=1.5,
            edgecolor=border_color,
            facecolor=bg_color,
            mutation_scale=1
        )
        ax.add_patch(rect)
        
        # Title
        ax.text(x + w/2, y + h - 0.2, title, fontsize=9.5, fontweight='bold', 
                ha='center', va='center', color=border_color)
        
        # Content text
        ax.text(x + w/2, y + h/2 - 0.1, text, fontsize=8.5, 
                ha='center', va='center', color=text_color, wrap=True)

    # Define arrow helper
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor='#64748b', edgecolor='#64748b', shrink=0.05, width=1.5, headwidth=6, headlength=6)
        )

    # Box dimensions and layout
    box_w = 2.4
    box_h = 1.2
    box_y = 0.6
    
    # Spacing between box starts: 2.4 width + 1.5 spacing = 3.9 step
    x1 = 0.2
    x2 = 4.1
    x3 = 8.0
    x4 = 11.9
    
    # 1. Chat CazéTV
    draw_box(x1, box_y, box_w, box_h, 
             "Chat CazéTV\n(Transmissões Esportivas)\n94k comentários brutos", 
             "FONTE DE DADOS", "#f8fafc", "#475569")
    
    # 2. Human Labelling
    draw_box(x2, box_y, box_w, box_h, 
             "Human Labelling\n(Consenso Absoluto)\n652 comentários", 
             "VALIDAÇÃO (GOLDEN SET)", "#eff6ff", "#2563eb")
    
    # 3. LLM 8B (Professor)
    draw_box(x3, box_y, box_w, box_h, 
             "LLM 8B (Professor)\n(Few-Shot + CoT Prompt)\nOtimizado contra Golden Set", 
             "TEACHER MODEL (LLM)", "#fffbeb", "#d97706")
    
    # 4. BERTimbau (Student)
    draw_box(x4, box_y, box_w, box_h, 
             "BERTimbau (Student)\n(110M parâmetros)\nFinetuning local (<5ms)", 
             "STUDENT MODEL (BERT)", "#faf5ff", "#9333ea")
    
    # Connections (Straight Arrows with wide spacing)
    arrow_y = box_y + box_h/2  # Center vertically
    
    # Chat CazéTV -> Human Labelling
    draw_arrow(x1 + box_w, arrow_y, x2, arrow_y)
    ax.text((x1 + box_w + x2)/2, arrow_y + 0.1, "Amostragem\n& Anotação", fontsize=8, color='#64748b', ha='center', va='bottom')
    
    # Human Labelling -> LLM 8B
    draw_arrow(x2 + box_w, arrow_y, x3, arrow_y)
    ax.text((x2 + box_w + x3)/2, arrow_y + 0.1, "Validação\n& Alinhamento", fontsize=8, color='#64748b', ha='center', va='bottom')
    
    # LLM 8B -> BERTimbau
    draw_arrow(x3 + box_w, arrow_y, x4, arrow_y)
    ax.text((x3 + box_w + x4)/2, arrow_y + 0.1, "Destilação\n& Fine-Tuning", fontsize=8, color='#64748b', ha='center', va='bottom')
    
    # Boundaries with padding
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 2.4)
    
    # Save flowchart
    os.makedirs('reports', exist_ok=True)
    plt.tight_layout()
    plt.savefig('reports/pipeline_flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Horizontal flowchart successfully created without clipping at reports/pipeline_flowchart.png")

if __name__ == "__main__":
    generate_flowchart()
