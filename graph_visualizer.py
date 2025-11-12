import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict
import json
import os


class GraphVisualizer:
    """Generates visualizations of knowledge graph connectivity and structure."""
    
    def __init__(self, graph: nx.Graph, output_dir: str = "graph_visualizations"):
        """
        Initialize visualizer.
        
        Args:
            graph: NetworkX graph object
            output_dir: Directory to save visualizations
        """
        self.graph = graph
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"📊 GraphVisualizer initialized, saving to {output_dir}/")
    
    def analyze_connectivity(self) -> Dict:
        """
        Analyze graph connectivity and component structure.
        
        Returns:
            Dictionary with connectivity metrics
        """
        print("\n🔍 Analyzing graph connectivity...")
        
        num_nodes = len(self.graph.nodes)
        num_edges = len(self.graph.edges)
        density = nx.density(self.graph)
        
        if nx.is_connected(self.graph):
            num_components = 1
            is_connected = True
        else:
            num_components = nx.number_connected_components(self.graph)
            is_connected = False
        
        components = list(nx.connected_components(self.graph))
        component_sizes = sorted([len(c) for c in components], reverse=True)
        
        # Centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness = nx.betweenness_centrality(self.graph)
        
        top_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
        
        analysis = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': round(density, 4),
            'is_connected': is_connected,
            'num_components': num_components,
            'component_sizes': component_sizes,
            'largest_component_size': component_sizes[0] if component_sizes else 0,
            'isolated_nodes': len([n for n in self.graph.nodes() if self.graph.degree(n) == 0]),
            'average_degree': 2 * num_edges / num_nodes if num_nodes > 0 else 0,
            'top_hubs': [(node, round(score, 4)) for node, score in top_hubs],
            'top_bridges': [(node, round(score, 4)) for node, score in top_bridges],
        }
        
        print(f"  ✓ Nodes: {num_nodes}, Edges: {num_edges}, Density: {density:.4f}")
        print(f"  ✓ Connected: {is_connected}, Components: {num_components}")
        print(f"  ✓ Largest component: {component_sizes[0]} nodes")
        
        return analysis
    
    def save_connectivity_report(self, analysis: Dict):
        """Save connectivity analysis to JSON."""
        report_path = os.path.join(self.output_dir, "connectivity_report.json")
        
        with open(report_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"✓ Connectivity report saved to {report_path}")
    
    def visualize_component_distribution(self):
        """Create bar chart of component sizes."""
        components = list(nx.connected_components(self.graph))
        component_sizes = sorted([len(c) for c in components], reverse=True)[:20]
        
        fig = px.bar(
            x=list(range(1, len(component_sizes) + 1)),
            y=component_sizes,
            labels={'x': 'Component Rank', 'y': 'Size (nodes)'},
            title='Knowledge Graph Component Distribution',
            color=component_sizes,
            color_continuous_scale='Blues'
        )
        
        fig_path = os.path.join(self.output_dir, "component_distribution.html")
        fig.write_html(fig_path)
        print(f"✓ Component distribution saved to {fig_path}")
    
    def visualize_degree_distribution(self):
        """Create degree distribution histogram."""
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        
        fig = px.histogram(
            degrees,
            nbins=50,
            labels={'value': 'Degree', 'count': 'Number of Nodes'},
            title='Degree Distribution - Knowledge Graph',
            color_discrete_sequence=['#1f77b4']
        )
        
        fig_path = os.path.join(self.output_dir, "degree_distribution.html")
        fig.write_html(fig_path)
        print(f"✓ Degree distribution saved to {fig_path}")
    
    def visualize_largest_component(self, max_nodes: int = 500):
        """Create interactive visualization of largest component."""
        components = list(nx.connected_components(self.graph))
        largest_component = max(components, key=len)
        
        subgraph = self.graph.subgraph(largest_component).copy()
        
        if len(subgraph.nodes) > max_nodes:
            print(f"⚠️  Largest component has {len(subgraph.nodes)} nodes, sampling {max_nodes}")
            # Sample for visualization
            import random
            sample_nodes = random.sample(list(subgraph.nodes), max_nodes)
            subgraph = subgraph.subgraph(sample_nodes).copy()
        
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
        
        edge_x = []
        edge_y = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            showlegend=False
        )
        
        node_x = []
        node_y = []
        node_color = []
        node_size = []
        node_text = []
        
        degree_dict = dict(subgraph.degree())
        
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            degree = degree_dict[node]
            node_size.append(10 + degree * 2)
            node_color.append(degree)
            
            concepts = self.graph.nodes[node].get('concepts', [])
            concept_text = ', '.join(concepts[:3]) if concepts else 'No concepts'
            node_text.append(f"Node {node}<br>Degree: {degree}<br>{concept_text}")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    thickness=15,
                    title='Node Degree',
                    len=0.7
                ),
                line=dict(width=1, color='white')
            )
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Largest Connected Component - Knowledge Graph',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700
            )
        )
        
        fig_path = os.path.join(self.output_dir, "largest_component.html")
        fig.write_html(fig_path)
        print(f"✓ Largest component visualization saved to {fig_path}")

    
    def create_connectivity_dashboard(self, analysis: Dict):
        """Create a summary dashboard."""
        
        summary_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Knowledge Graph Connectivity Dashboard</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .metric {{ background: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric-title {{ font-weight: bold; color: #333; font-size: 14px; }}
                .metric-value {{ font-size: 24px; color: #1f77b4; font-weight: bold; margin-top: 5px; }}
                .section {{ margin: 20px 0; }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; border-bottom: 2px solid #1f77b4; padding-bottom: 5px; }}
                table {{ width: 100%; border-collapse: collapse; background: white; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #1f77b4; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                a {{ color: #1f77b4; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <h1>⚽ Knowledge Graph Connectivity Dashboard</h1>
            
            <div class="section">
                <h2>Graph Overview</h2>
                <div class="metric">
                    <div class="metric-title">Total Nodes</div>
                    <div class="metric-value">{analysis['num_nodes']}</div>
                </div>
                <div class="metric">
                    <div class="metric-title">Total Edges</div>
                    <div class="metric-value">{analysis['num_edges']}</div>
                </div>
                <div class="metric">
                    <div class="metric-title">Graph Density</div>
                    <div class="metric-value">{analysis['density']}</div>
                </div>
                <div class"metric">
                    <div class="metric-title">Average Node Degree</div>
                    <div class="metric-value">{analysis['average_degree']:.2f}</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Connectivity Status</h2>
                <div class="metric">
                    <div class="metric-title">Is Connected</div>
                    <div class="metric-value">{"✅ Yes" if analysis['is_connected'] else "❌ No"}</div>
                </div>
                <div class="metric">
                    <div class="metric-title">Connected Components</div>
                    <div class="metric-value">{analysis['num_components']}</div>
                </div>
                <div class="metric">
                    <div class="metric-title">Largest Component Size</div>
                    <div class="metric-value">{analysis['largest_component_size']} nodes ({analysis['largest_component_size']/analysis['num_nodes']*100:.1f}%)</div>
                </div>
                <div class="metric">
                    <div class="metric-title">Isolated Nodes</div>
                    <div class="metric-value">{analysis['isolated_nodes']}</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Top Hub Nodes (High Degree Centrality)</h2>
                <table>
                    <tr><th>Node ID</th><th>Degree Centrality</th></tr>
                    {''.join(f'<tr><td>{node}</td><td>{score}</td></tr>' for node, score in analysis['top_hubs'])}
                </table>
            </div>
            
            <div class="section">
                <h2>Top Bridge Nodes (High Betweenness)</h2>
                <table>
                    <tr><th>Node ID</th><th>Betweenness Centrality</th></tr>
                    {''.join(f'<tr><td>{node}</td><td>{score}</td></tr>' for node, score in analysis['top_bridges'])}
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <ul>
                    <li><a href="largest_component.html">Interactive Largest Component</a></li>
                    <li><a href="degree_distribution.html">Degree Distribution</a></li>
                    <li><a href="component_distribution.html">Component Distribution</a></li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        dashboard_path = os.path.join(self.output_dir, "dashboard.html")
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(summary_html)
        
        print(f"✓ Dashboard saved to {dashboard_path}")

    
    def generate_all_visualizations(self):
        """Generate all visualizations and reports."""
        print("\n" + "="*60)
        print("📊 Generating Graph Visualizations")
        print("="*60)
        
        analysis = self.analyze_connectivity()
        self.save_connectivity_report(analysis)
        self.visualize_degree_distribution()
        self.visualize_component_distribution()
        self.visualize_largest_component()
        self.create_connectivity_dashboard(analysis)
        
        print("\n" + "="*60)
        print("✅ All visualizations generated!")
        print(f"📂 Open: graph_visualizations/dashboard.html")
        print("="*60)
