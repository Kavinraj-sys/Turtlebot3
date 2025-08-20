#!/usr/bin/env python3
"""
Custom A* and Dijkstra Path Planning Algorithms for TurtleBot3
This implementation works with occupancy grid maps from ROS2
Enhanced with path length and time measurements
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import heapq
import math
import time
from typing import List, Tuple, Optional, Dict

class PathMetrics:
    """Class to store path planning metrics"""
    def __init__(self, algorithm: str, path_length: float, planning_time: float, 
                 nodes_explored: int, path_points: int):
        self.algorithm = algorithm
        self.path_length = path_length
        self.planning_time = planning_time
        self.nodes_explored = nodes_explored
        self.path_points = path_points

class PathPlanningNode(Node):
    def __init__(self):
        super().__init__('path_planning_node')
        
        # Publishers
        self.path_publisher = self.create_publisher(Path, '/planned_path', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/path_markers', 10)
        
        # Subscribers
        self.map_subscriber = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.goal_subscriber = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)
        
        # Variables
        self.map_data = None
        self.robot_pose = None
        self.current_algorithm = "astar"  # or "dijkstra"
        self.last_metrics = None  # Store metrics from last planning
        
        self.get_logger().info('Path Planning Node initialized')

    def map_callback(self, msg: OccupancyGrid):
        """Store the occupancy grid map"""
        self.map_data = msg
        self.get_logger().info('Map received')

    def goal_callback(self, msg: PoseStamped):
        """Plan path when goal is received"""
        if self.map_data is None:
            self.get_logger().warn('No map available for planning')
            return
        
        # For demo, assume robot starts at origin
        start = (0.0, 0.0)
        goal = (msg.pose.position.x, msg.pose.position.y)
        
        self.get_logger().info(f'Planning path from {start} to {goal}')
        
        # Plan path using selected algorithm
        if self.current_algorithm == "astar":
            path, metrics = self.plan_astar(start, goal)
        else:
            path, metrics = self.plan_dijkstra(start, goal)
        
        if path and metrics:
            self.last_metrics = metrics
            self.publish_path(path)
            self.visualize_path(path)
            self.log_metrics(metrics)
        else:
            self.get_logger().warn('Path planning failed')

    def calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Calculate the total Euclidean distance of the path"""
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            total_length += math.sqrt(dx * dx + dy * dy)
        
        return total_length

    def log_metrics(self, metrics: PathMetrics):
        """Log detailed path planning metrics"""
        self.get_logger().info(f'📊 PATH PLANNING METRICS ({metrics.algorithm.upper()})')
        self.get_logger().info(f'   ⏱️  Planning Time: {metrics.planning_time:.4f} seconds')
        self.get_logger().info(f'   📏 Path Length: {metrics.path_length:.4f} meters')
        self.get_logger().info(f'   🔍 Nodes Explored: {metrics.nodes_explored}')
        self.get_logger().info(f'   📍 Path Points: {metrics.path_points}')
        self.get_logger().info(f'   🚀 Avg Speed Required: {metrics.path_length/max(metrics.planning_time, 0.001):.2f} m/s (if completed in planning time)')

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        if self.map_data is None:
            return (0, 0)
        
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        resolution = self.map_data.info.resolution
        
        grid_x = int((x - origin_x) / resolution)
        grid_y = int((y - origin_y) / resolution)
        
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        if self.map_data is None:
            return (0.0, 0.0)
        
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        resolution = self.map_data.info.resolution
        
        x = origin_x + (grid_x + 0.5) * resolution
        y = origin_y + (grid_y + 0.5) * resolution
        
        return (x, y)

    def is_valid_cell(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid cell is valid and not occupied"""
        if self.map_data is None:
            return False
        
        width = self.map_data.info.width
        height = self.map_data.info.height
        
        if grid_x < 0 or grid_x >= width or grid_y < 0 or grid_y >= height:
            return False
        
        index = grid_y * width + grid_x
        if index >= len(self.map_data.data):
            return False
        
        # Check if cell is free (0 = free, 100 = occupied, -1 = unknown)
        cell_value = self.map_data.data[index]
        return cell_value >= 0 and cell_value < 50  # Allow free and low-probability cells

    def get_neighbors(self, grid_x: int, grid_y: int) -> List[Tuple[int, int, float]]:
        """Get valid neighbors with movement costs"""
        neighbors = []
        
        # 8-connected neighbors (include diagonals)
        directions = [
            (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
            (0, -1, 1.0),                         (0, 1, 1.0),
            (1, -1, math.sqrt(2)),  (1, 0, 1.0),  (1, 1, math.sqrt(2))
        ]
        
        for dx, dy, cost in directions:
            new_x, new_y = grid_x + dx, grid_y + dy
            if self.is_valid_cell(new_x, new_y):
                neighbors.append((new_x, new_y, cost))
        
        return neighbors

    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Euclidean distance heuristic for A*"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)

    def plan_astar(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Tuple[Optional[List[Tuple[float, float]]], Optional[PathMetrics]]:
        """A* path planning algorithm with metrics"""
        start_time = time.perf_counter()  # More precise timing
        
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])
        
        if not self.is_valid_cell(start_grid[0], start_grid[1]):
            self.get_logger().error('Start position is not valid')
            return None, None
        
        if not self.is_valid_cell(goal_grid[0], goal_grid[1]):
            self.get_logger().error('Goal position is not valid')
            return None, None
        
        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        
        explored_nodes = 0
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            explored_nodes += 1
            
            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    world_pos = self.grid_to_world(current[0], current[1])
                    path.append(world_pos)
                    current = came_from[current]
                
                # Add start position
                path.append(start)
                path.reverse()
                
                planning_time = time.perf_counter() - start_time
                path_length = self.calculate_path_length(path)
                
                metrics = PathMetrics(
                    algorithm="A*",
                    path_length=path_length,
                    planning_time=planning_time,
                    nodes_explored=explored_nodes,
                    path_points=len(path)
                )
                
                return path, metrics
            
            for neighbor_x, neighbor_y, move_cost in self.get_neighbors(current[0], current[1]):
                neighbor = (neighbor_x, neighbor_y)
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)
                    
                    # Add to open set if not already there
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        self.get_logger().error('A* failed to find path')
        return None, None

    def plan_dijkstra(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Tuple[Optional[List[Tuple[float, float]]], Optional[PathMetrics]]:
        """Dijkstra's path planning algorithm with metrics"""
        start_time = time.perf_counter()  # More precise timing
        
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])
        
        if not self.is_valid_cell(start_grid[0], start_grid[1]):
            self.get_logger().error('Start position is not valid')
            return None, None
        
        if not self.is_valid_cell(goal_grid[0], goal_grid[1]):
            self.get_logger().error('Goal position is not valid')
            return None, None
        
        # Dijkstra's algorithm
        distances = {start_grid: 0}
        came_from = {}
        unvisited = [(0, start_grid)]
        visited = set()
        
        explored_nodes = 0
        
        while unvisited:
            current_distance, current = heapq.heappop(unvisited)
            
            if current in visited:
                continue
            
            visited.add(current)
            explored_nodes += 1
            
            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    world_pos = self.grid_to_world(current[0], current[1])
                    path.append(world_pos)
                    current = came_from[current]
                
                # Add start position
                path.append(start)
                path.reverse()
                
                planning_time = time.perf_counter() - start_time
                path_length = self.calculate_path_length(path)
                
                metrics = PathMetrics(
                    algorithm="Dijkstra",
                    path_length=path_length,
                    planning_time=planning_time,
                    nodes_explored=explored_nodes,
                    path_points=len(path)
                )
                
                return path, metrics
            
            for neighbor_x, neighbor_y, move_cost in self.get_neighbors(current[0], current[1]):
                neighbor = (neighbor_x, neighbor_y)
                
                if neighbor in visited:
                    continue
                
                new_distance = current_distance + move_cost
                
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    came_from[neighbor] = current
                    heapq.heappush(unvisited, (new_distance, neighbor))
        
        self.get_logger().error('Dijkstra failed to find path')
        return None, None

    def publish_path(self, path: List[Tuple[float, float]]):
        """Publish the planned path"""
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for x, y in path:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_publisher.publish(path_msg)

    def visualize_path(self, path: List[Tuple[float, float]]):
        """Visualize the path with markers"""
        marker_array = MarkerArray()
        
        # Path line marker
        line_marker = Marker()
        line_marker.header.frame_id = 'map'
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = 'path'
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.05
        line_marker.color.r = 1.0
        line_marker.color.g = 0.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0
        
        for x, y in path:
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.1
            line_marker.points.append(point)
        
        marker_array.markers.append(line_marker)
        
        # Waypoint markers
        for i, (x, y) in enumerate(path[::5]):  # Show every 5th waypoint
            waypoint_marker = Marker()
            waypoint_marker.header.frame_id = 'map'
            waypoint_marker.header.stamp = self.get_clock().now().to_msg()
            waypoint_marker.ns = 'waypoints'
            waypoint_marker.id = i
            waypoint_marker.type = Marker.SPHERE
            waypoint_marker.action = Marker.ADD
            waypoint_marker.pose.position.x = x
            waypoint_marker.pose.position.y = y
            waypoint_marker.pose.position.z = 0.1
            waypoint_marker.pose.orientation.w = 1.0
            waypoint_marker.scale.x = 0.1
            waypoint_marker.scale.y = 0.1
            waypoint_marker.scale.z = 0.1
            waypoint_marker.color.r = 0.0
            waypoint_marker.color.g = 1.0
            waypoint_marker.color.b = 0.0
            waypoint_marker.color.a = 1.0
            
            marker_array.markers.append(waypoint_marker)
        
        self.marker_publisher.publish(marker_array)

    def set_algorithm(self, algorithm: str):
        """Switch between A* and Dijkstra algorithms"""
        if algorithm.lower() in ['astar', 'a*', 'a_star']:
            self.current_algorithm = "astar"
            self.get_logger().info('Switched to A* algorithm')
        elif algorithm.lower() in ['dijkstra', 'dijkstra\'s']:
            self.current_algorithm = "dijkstra"
            self.get_logger().info('Switched to Dijkstra algorithm')
        else:
            self.get_logger().warn(f'Unknown algorithm: {algorithm}')


class InteractivePathPlanner(Node):
    """Interactive interface for testing path planning algorithms"""
    
    def __init__(self):
        super().__init__('interactive_path_planner')
        
        # Create path planning node
        self.planner = PathPlanningNode()
        
        # Publisher for goal poses
        self.goal_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # Store metrics for comparison
        self.comparison_metrics = {}
        
        self.get_logger().info('Interactive Path Planner ready!')

    def send_goal(self, x: float, y: float, algorithm: str = "astar"):
        """Send a goal and specify which algorithm to use"""
        # Set algorithm
        self.planner.set_algorithm(algorithm)
        
        # Create and publish goal
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = float(x)
        goal_msg.pose.position.y = float(y)
        goal_msg.pose.orientation.w = 1.0
        
        self.goal_publisher.publish(goal_msg)
        self.get_logger().info(f'Sent goal ({x}, {y}) using {algorithm.upper()}')
        
        # Wait a bit for processing and store metrics
        time.sleep(0.5)
        if self.planner.last_metrics:
            self.comparison_metrics[algorithm] = self.planner.last_metrics

    def compare_algorithms(self, x: float, y: float):
        """Compare A* and Dijkstra algorithms"""
        print(f"\n🔄 ALGORITHM COMPARISON for goal ({x}, {y})")
        print("=" * 60)
        
        # Clear previous comparison data
        self.comparison_metrics = {}
        
        # Run A*
        print("🔍 Running A*...")
        self.send_goal(x, y, "astar")
        time.sleep(1)
        
        # Run Dijkstra
        print("🔍 Running Dijkstra...")
        self.send_goal(x, y, "dijkstra")
        time.sleep(1)
        
        # Compare results
        if len(self.comparison_metrics) == 2:
            astar_metrics = self.comparison_metrics.get("astar")
            dijkstra_metrics = self.comparison_metrics.get("dijkstra")
            
            print("\n📊 COMPARISON RESULTS:")
            print("-" * 60)
            print(f"{'Metric':<20} {'A*':<15} {'Dijkstra':<15} {'Winner':<10}")
            print("-" * 60)
            
            # Planning time comparison
            if astar_metrics.planning_time < dijkstra_metrics.planning_time:
                time_winner = "A*"
            elif dijkstra_metrics.planning_time < astar_metrics.planning_time:
                time_winner = "Dijkstra"
            else:
                time_winner = "Tie"
            
            print(f"{'Planning Time (s)':<20} {astar_metrics.planning_time:<15.4f} "
                  f"{dijkstra_metrics.planning_time:<15.4f} {time_winner:<10}")
            
            # Path length comparison
            if astar_metrics.path_length < dijkstra_metrics.path_length:
                length_winner = "A*"
            elif dijkstra_metrics.path_length < astar_metrics.path_length:
                length_winner = "Dijkstra"
            else:
                length_winner = "Tie"
            
            print(f"{'Path Length (m)':<20} {astar_metrics.path_length:<15.4f} "
                  f"{dijkstra_metrics.path_length:<15.4f} {length_winner:<10}")
            
            # Nodes explored comparison
            if astar_metrics.nodes_explored < dijkstra_metrics.nodes_explored:
                nodes_winner = "A*"
            elif dijkstra_metrics.nodes_explored < astar_metrics.nodes_explored:
                nodes_winner = "Dijkstra"
            else:
                nodes_winner = "Tie"
            
            print(f"{'Nodes Explored':<20} {astar_metrics.nodes_explored:<15} "
                  f"{dijkstra_metrics.nodes_explored:<15} {nodes_winner:<10}")
            
            # Path points comparison
            if astar_metrics.path_points < dijkstra_metrics.path_points:
                points_winner = "A*"
            elif dijkstra_metrics.path_points < astar_metrics.path_points:
                points_winner = "Dijkstra"
            else:
                points_winner = "Tie"
            
            print(f"{'Path Points':<20} {astar_metrics.path_points:<15} "
                  f"{dijkstra_metrics.path_points:<15} {points_winner:<10}")
            
            print("-" * 60)
            
            # Calculate efficiency metrics
            astar_efficiency = astar_metrics.path_length / (astar_metrics.planning_time * astar_metrics.nodes_explored)
            dijkstra_efficiency = dijkstra_metrics.path_length / (dijkstra_metrics.planning_time * dijkstra_metrics.nodes_explored)
            
            print(f"\n🎯 EFFICIENCY ANALYSIS:")
            print(f"A* Efficiency Score: {astar_efficiency:.6f}")
            print(f"Dijkstra Efficiency Score: {dijkstra_efficiency:.6f}")
            
            if astar_efficiency > dijkstra_efficiency:
                print("🏆 A* is more efficient overall")
            elif dijkstra_efficiency > astar_efficiency:
                print("🏆 Dijkstra is more efficient overall")
            else:
                print("🤝 Both algorithms have similar efficiency")


def main():
    rclpy.init()
    
    print("🔧 Enhanced Path Planning with Metrics Analysis")
    print("=" * 55)
    
    # Create nodes
    planner = PathPlanningNode()
    interactive = InteractivePathPlanner()
    
    print("📋 Commands:")
    print("  - 'astar x y' - Plan path to (x,y) using A*")
    print("  - 'dijkstra x y' - Plan path to (x,y) using Dijkstra")
    print("  - 'compare x y' - Compare both algorithms with detailed metrics")
    print("  - 'quit' - Exit")
    print("\nMake sure RViz is open to visualize the planned paths!")
    print("Metrics include: planning time, path length, nodes explored, and efficiency scores")
    
    import threading
    
    # Run ROS nodes in separate thread
    def spin_nodes():
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(planner)
        executor.add_node(interactive)
        executor.spin()
    
    spinner_thread = threading.Thread(target=spin_nodes)
    spinner_thread.daemon = True
    spinner_thread.start()
    
    try:
        while True:
            user_input = input("\n🎯 Enter command: ").strip().split()
            
            if not user_input:
                continue
                
            command = user_input[0].lower()
            
            if command == 'quit':
                break
            elif command in ['astar', 'a*'] and len(user_input) >= 3:
                try:
                    x, y = float(user_input[1]), float(user_input[2])
                    interactive.send_goal(x, y, "astar")
                except ValueError:
                    print("❌ Invalid coordinates. Use: astar x y")
            elif command == 'dijkstra' and len(user_input) >= 3:
                try:
                    x, y = float(user_input[1]), float(user_input[2])
                    interactive.send_goal(x, y, "dijkstra")
                except ValueError:
                    print("❌ Invalid coordinates. Use: dijkstra x y")
            elif command == 'compare' and len(user_input) >= 3:
                try:
                    x, y = float(user_input[1]), float(user_input[2])
                    interactive.compare_algorithms(x, y)
                except ValueError:
                    print("❌ Invalid coordinates. Use: compare x y")
            else:
                print("❌ Invalid command. Available commands:")
                print("   astar x y, dijkstra x y, compare x y, quit")
                
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    
    planner.destroy_node()
    interactive.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()