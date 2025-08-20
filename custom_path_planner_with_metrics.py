#!/usr/bin/env python3
"""
Custom A* and Dijkstra Path Planning Algorithms for TurtleBot3
This implementation works with occupancy grid maps from ROS2
Enhanced with path length and time metrics
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
from typing import List, Tuple, Optional

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
        self.last_planning_metrics = {}
        
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
            path = self.plan_astar(start, goal)
        else:
            path = self.plan_dijkstra(start, goal)
        
        if path:
            self.publish_path(path)
            self.visualize_path(path)
            self.get_logger().info(f'Path planned with {len(path)} waypoints')
        else:
            self.get_logger().warn('Path planning failed')

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

    def calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Calculate the total length of the path in meters"""
        if not path or len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            segment_length = math.sqrt(dx * dx + dy * dy)
            total_length += segment_length
        
        return total_length

    def plan_astar(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """A* path planning algorithm with path metrics"""
        start_time = time.time()
        
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])
        
        if not self.is_valid_cell(start_grid[0], start_grid[1]):
            self.get_logger().error('Start position is not valid')
            return None
        
        if not self.is_valid_cell(goal_grid[0], goal_grid[1]):
            self.get_logger().error('Goal position is not valid')
            return None
        
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
                
                # Calculate metrics
                planning_time = time.time() - start_time
                path_length = self.calculate_path_length(path)
                
                self.get_logger().info(f'A* completed: {explored_nodes} nodes explored, '
                                     f'{planning_time:.3f}s, path length: {len(path)} waypoints, '
                                     f'distance: {path_length:.3f}m')
                
                # Store metrics for comparison
                self.last_planning_metrics = {
                    'algorithm': 'A*',
                    'planning_time': planning_time,
                    'path_length': path_length,
                    'waypoints': len(path),
                    'nodes_explored': explored_nodes
                }
                
                return path
            
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
        return None

    def plan_dijkstra(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """Dijkstra's path planning algorithm with path metrics"""
        start_time = time.time()
        
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])
        
        if not self.is_valid_cell(start_grid[0], start_grid[1]):
            self.get_logger().error('Start position is not valid')
            return None
        
        if not self.is_valid_cell(goal_grid[0], goal_grid[1]):
            self.get_logger().error('Goal position is not valid')
            return None
        
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
                
                # Calculate metrics
                planning_time = time.time() - start_time
                path_length = self.calculate_path_length(path)
                
                self.get_logger().info(f'Dijkstra completed: {explored_nodes} nodes explored, '
                                     f'{planning_time:.3f}s, path length: {len(path)} waypoints, '
                                     f'distance: {path_length:.3f}m')
                
                # Store metrics for comparison
                self.last_planning_metrics = {
                    'algorithm': 'Dijkstra',
                    'planning_time': planning_time,
                    'path_length': path_length,
                    'waypoints': len(path),
                    'nodes_explored': explored_nodes
                }
                
                return path
            
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
        return None

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

    def get_path_metrics(self) -> dict:
        """Get the metrics from the last planned path"""
        return self.last_planning_metrics.copy() if self.last_planning_metrics else {}

    def print_path_comparison(self, metrics_list: List[dict]):
        """Print comparison between different algorithm results"""
        print("\n📊 Path Planning Comparison:")
        print("-" * 60)
        print(f"{'Algorithm':<12} {'Time(s)':<8} {'Distance(m)':<12} {'Waypoints':<10} {'Nodes':<8}")
        print("-" * 60)
        
        for metrics in metrics_list:
            print(f"{metrics['algorithm']:<12} "
                  f"{metrics['planning_time']:<8.3f} "
                  f"{metrics['path_length']:<12.3f} "
                  f"{metrics['waypoints']:<10} "
                  f"{metrics['nodes_explored']:<8}")
        
        if len(metrics_list) == 2:
            print("-" * 60)
            time_diff = metrics_list[1]['planning_time'] - metrics_list[0]['planning_time']
            dist_diff = metrics_list[1]['path_length'] - metrics_list[0]['path_length']
            print(f"Time difference: {time_diff:+.3f}s")
            print(f"Distance difference: {dist_diff:+.3f}m")


class InteractivePathPlanner(Node):
    """Interactive interface for testing path planning algorithms"""
    
    def __init__(self):
        super().__init__('interactive_path_planner')
        
        # Create path planning node
        self.planner = PathPlanningNode()
        
        # Publisher for goal poses
        self.goal_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        self.get_logger().info('Interactive Path Planner ready!')

    def create_goal_message(self, x: float, y: float) -> PoseStamped:
        """Helper to create goal message"""
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = float(x)
        goal_msg.pose.position.y = float(y)
        goal_msg.pose.orientation.w = 1.0
        return goal_msg

    def send_goal(self, x: float, y: float, algorithm: str = "astar"):
        """Send a goal and specify which algorithm to use"""
        # Set algorithm
        self.planner.set_algorithm(algorithm)
        
        # Create and publish goal
        goal_msg = self.create_goal_message(x, y)
        self.goal_publisher.publish(goal_msg)
        self.get_logger().info(f'Sent goal ({x}, {y}) using {algorithm.upper()}')

    def enhanced_compare_goal(self, x: float, y: float):
        """Compare both algorithms and show detailed metrics"""
        metrics_list = []
        
        print(f"🔄 Comparing A* and Dijkstra for goal ({x}, {y})")
        
        # Test A*
        print("Running A*...")
        self.planner.set_algorithm("astar")
        goal_msg = self.create_goal_message(x, y)
        self.goal_publisher.publish(goal_msg)
        time.sleep(2)  # Wait for processing
        astar_metrics = self.planner.get_path_metrics()
        if astar_metrics:
            metrics_list.append(astar_metrics)
        
        # Test Dijkstra
        print("Running Dijkstra...")
        self.planner.set_algorithm("dijkstra")
        self.goal_publisher.publish(goal_msg)
        time.sleep(2)  # Wait for processing
        dijkstra_metrics = self.planner.get_path_metrics()
        if dijkstra_metrics:
            metrics_list.append(dijkstra_metrics)
        
        # Print comparison
        if len(metrics_list) == 2:
            self.planner.print_path_comparison(metrics_list)
        elif len(metrics_list) == 1:
            print("⚠️  Only one algorithm completed successfully")
        else:
            print("❌ Both algorithms failed to find a path")


def main():
    rclpy.init()
    
    print("🔧 Custom Path Planning with A* and Dijkstra")
    print("=" * 50)
    
    # Create nodes
    interactive = InteractivePathPlanner()
    
    print("📋 Commands:")
    print("  - 'astar x y' - Plan path to (x,y) using A*")
    print("  - 'dijkstra x y' - Plan path to (x,y) using Dijkstra")
    print("  - 'compare x y' - Compare both algorithms")
    print("  - 'metrics' - Show last planning metrics")
    print("  - 'quit' - Exit")
    print("\nMake sure RViz is open to visualize the planned paths!")
    
    import threading
    
    # Run ROS nodes in separate thread
    def spin_nodes():
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(interactive.planner)
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
                    interactive.enhanced_compare_goal(x, y)
                except ValueError:
                    print("❌ Invalid coordinates. Use: compare x y")
            elif command == 'metrics':
                metrics = interactive.planner.get_path_metrics()
                if metrics:
                    print("\n📊 Last Planning Metrics:")
                    print(f"Algorithm: {metrics['algorithm']}")
                    print(f"Planning Time: {metrics['planning_time']:.3f}s")
                    print(f"Path Length: {metrics['path_length']:.3f}m")
                    print(f"Waypoints: {metrics['waypoints']}")
                    print(f"Nodes Explored: {metrics['nodes_explored']}")
                else:
                    print("❌ No metrics available. Plan a path first.")
            else:
                print("❌ Invalid command. Available commands:")
                print("   astar x y, dijkstra x y, compare x y, metrics, quit")
                
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    
    interactive.planner.destroy_node()
    interactive.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()