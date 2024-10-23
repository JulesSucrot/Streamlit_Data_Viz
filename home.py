import streamlit as st
import time
import numpy as np
import streamlit.components.v1 as components

def progressive_write(text):
    for i in text:
        yield i
        time.sleep(0.05)



def app():
    st.title('Jules Sucrot')
    st.write_stream(progressive_write(f'## Welcome to my streamlit resume!'))

    with st.container(border=True):
        import plotly.graph_objects as go

        # Create the figure
        fig = go.Figure()

        # Add the body (vertical line)
        fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=2, line=dict(color="black", width=4))

        # Add the head (circle)
        fig.add_shape(type="circle", x0=-0.25, y0=2, x1=0.25, y1=2.75, line=dict(color="black", width=4))

        # Add the arms (horizontal line)
        fig.add_shape(type="line", x0=-0.25, y0=0.75, x1=0, y1=1.5, line=dict(color="black", width=4))

        # Add the left leg (diagonal line)
        fig.add_shape(type="line", x0=0, y0=0, x1=-0.5, y1=-1, line=dict(color="black", width=4))

        # Add the right leg (diagonal line)
        fig.add_shape(type="line", x0=0, y0=0, x1=0.5, y1=-1, line=dict(color="black", width=4))

        # Define frames for waving the right arm (upward motion)
        frames = []
        angles_up = np.linspace(0, np.pi/4, 30)  # Arm moves up from horizontal to ~45 degrees

        # Create upward motion frames
        for angle in angles_up:
            x1 = 0.6 * np.cos(angle)
            y1 = 1.5 + np.sin(angle)
            
            frame = go.Frame(
                data=[go.Scatter(x=[0, x1], y=[1.5, y1], mode='lines', line=dict(color='black', width=4))]
            )
            frames.append(frame)

        # Add the first arm position (horizontal) as the static trace
        fig.add_trace(go.Scatter(x=[0, 0.6], y=[1.5, 1.5], mode='lines', line=dict(color='black', width=4)))

        # Add animation frames
        frames+=frames[::-1]
        frames+=frames*2
        fig.frames = frames

        # Set up layout with animation
        fig.update_layout(
            title="Welcome",
            xaxis=dict(range=[-2, 2], zeroline=False),
            yaxis=dict(range=[-2, 3], zeroline=False),
            showlegend=False,
            height=500,
            width=500,
            updatemenus=[dict(type="buttons",
                            buttons=[dict(label="Play",
                                            method="animate",
                                            args=[None, {"frame": {"duration": 2, "redraw": True},
                                                        "fromcurrent": True,
                                                        "mode": "immediate",
                                                        "loop": True}]),
                                    dict(label="Pause",
                                            method="animate",
                                            args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                        "mode": "immediate"}])])]
        )
        

        st.plotly_chart(fig)

app()