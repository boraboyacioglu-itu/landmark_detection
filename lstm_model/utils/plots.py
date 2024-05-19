from plotly import graph_objs as go

def plot_fig(lm):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=lm[0],
        y=lm[1],
        z=lm[2],
        mode='markers',
        marker=dict(size=3, color='red'),
        name='3D Landmarks'
    ))

    # Update layout for interactive controls
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='cube',  # Cube aspect ratio for equal scaling
            camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=0, y=0, z=-2)),  # Initial camera position
        ),
    )
    
    return fig