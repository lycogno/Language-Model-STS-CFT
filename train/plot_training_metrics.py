import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_training_metrics(checkpoint_dir):
    """Plot training metrics from trainer_state.json"""
    
    # Find the trainer_state.json file
    checkpoint_path = Path(checkpoint_dir)
    trainer_state_file = checkpoint_path / "trainer_state.json"
    
    if not trainer_state_file.exists():
        print(f"Error: {trainer_state_file} not found")
        return
    
    # Load the training state
    with open(trainer_state_file, 'r') as f:
        trainer_state = json.load(f)
    
    # Extract metrics from log_history
    log_history = trainer_state.get('log_history', [])
    
    steps = []
    losses = []
    learning_rates = []
    
    for entry in log_history:
        if 'loss' in entry:
            steps.append(entry['step'])
            losses.append(entry['loss'])
            if 'learning_rate' in entry:
                learning_rates.append(entry['learning_rate'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot training loss
    axes[0].plot(steps, losses, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel('Training Steps', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss over Steps', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(left=0)
    
    # Add text with final metrics
    if losses:
        final_loss = losses[-1]
        initial_loss = losses[0]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        
        textstr = f'Initial Loss: {initial_loss:.4f}\nFinal Loss: {final_loss:.4f}\nImprovement: {improvement:.2f}%'
        axes[0].text(0.02, 0.98, textstr, transform=axes[0].transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot learning rate
    if learning_rates:
        axes[1].plot(steps, learning_rates, 'g-', linewidth=2, marker='s', markersize=4)
        axes[1].set_xlabel('Training Steps', fontsize=12)
        axes[1].set_ylabel('Learning Rate', fontsize=12)
        axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(left=0)
        axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = checkpoint_path / "training_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Total Steps: {steps[-1] if steps else 0}")
    print(f"Initial Loss: {losses[0]:.4f}" if losses else "N/A")
    print(f"Final Loss: {losses[-1]:.4f}" if losses else "N/A")
    if len(losses) > 1:
        improvement = ((losses[0] - losses[-1]) / losses[0]) * 100
        print(f"Loss Improvement: {improvement:.2f}%")
        print(f"Average Loss: {sum(losses)/len(losses):.4f}")
        print(f"Min Loss: {min(losses):.4f} at step {steps[losses.index(min(losses))]}")
    if learning_rates:
        print(f"Initial LR: {learning_rates[0]:.2e}")
        print(f"Final LR: {learning_rates[-1]:.2e}")
    print("="*50)
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training metrics from trainer_state.json')
    parser.add_argument('checkpoint_dir', type=str, 
                       help='Path to checkpoint directory containing trainer_state.json')
    
    args = parser.parse_args()
    plot_training_metrics(args.checkpoint_dir)
