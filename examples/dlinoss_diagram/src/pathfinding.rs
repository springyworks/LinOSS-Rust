//! Temporary pathfinding logic for D-LinOSS diagrams.
//! This module contains the original, heuristic-based routing algorithms.
//! It is intended to be replaced by the `maxGraphRust` crate once it is ready.

use egui::{Color32, Pos2, Rect, Vec2};
use crate::diagram_renderer_simple::DraggableComponent;

// Calculate optimal right-angled path between two rectangles
pub fn calculate_optimal_right_angle_path(
    from: Rect,
    to: Rect,
    from_id: &str,
    to_id: &str,
    components: &[DraggableComponent],
    zoom: f32,
) -> Vec<Pos2> {
    let from_center = from.center();
    let to_center = to.center();
    let edge_clearance = 10.0 * zoom; // Increased clearance

    // Determine primary direction
    let dx = to_center.x - from_center.x;
    let dy = to_center.y - from_center.y;

    // Connection points on edges, now with clearance
    let (start, end) = if dx.abs() > dy.abs() {
        // Horizontal connection
        if dx > 0.0 {
            // Left to right
            (
                egui::pos2(from.right() + edge_clearance, from_center.y),
                egui::pos2(to.left() - edge_clearance, to_center.y),
            )
        } else {
            // Right to left
            (
                egui::pos2(from.left() - edge_clearance, from_center.y),
                egui::pos2(to.right() + edge_clearance, to_center.y),
            )
        }
    } else {
        // Vertical connection
        if dy > 0.0 {
            // Top to bottom
            (
                egui::pos2(from_center.x, from.bottom() + edge_clearance),
                egui::pos2(to.center().x, to.top() - edge_clearance),
            )
        } else {
            // Bottom to top
            (
                egui::pos2(from_center.x, from.top() - edge_clearance),
                egui::pos2(to.center().x, to.bottom() + edge_clearance),
            )
        }
    };

    // Special handling for feedback connections (A to oscillator)
    if from_id == "feedback" && to_id == "oscillator" {
        return create_feedback_path(from, to, components, zoom);
    }

    // If components are primarily horizontal, attempt a Z-shaped path first.
    if dx.abs() > dy.abs() {
        let mid_x = from_center.x + dx / 2.0;
        let z_path = vec![
            start,
            egui::pos2(mid_x, start.y),
            egui::pos2(mid_x, end.y),
            end,
        ];
        if is_path_clear(&z_path, from_id, to_id, components, zoom) {
            return z_path;
        }
    } else {
        // Primarily vertical, attempt an N-shaped path
        let mid_y = from_center.y + dy / 2.0;
        let n_path = vec![
            start,
            egui::pos2(start.x, mid_y),
            egui::pos2(end.x, mid_y),
            end,
        ];
        if is_path_clear(&n_path, from_id, to_id, components, zoom) {
            return n_path;
        }
    }

    // Try simple L-shaped path
    let simple_path = vec![start, egui::pos2(end.x, start.y), end];

    if is_path_clear(&simple_path, from_id, to_id, components, zoom) {
        return simple_path;
    }

    // Try alternative L-shaped path
    let alt_path = vec![start, egui::pos2(start.x, end.y), end];

    if is_path_clear(&alt_path, from_id, to_id, components, zoom) {
        return alt_path;
    }

    // Create bypass path with clearance
    create_bypass_path(start, end, from, to, components, zoom)
}

// Create a feedback loop path that goes around components
fn create_feedback_path(from: Rect, to: Rect, components: &[DraggableComponent], zoom: f32) -> Vec<Pos2> {
    let clearance = 40.0 * zoom;
    let edge_clearance = 10.0 * zoom; // Increased clearance

    // Find the lowest component to route below it
    let mut max_y = from.bottom().max(to.bottom());
    for comp in components {
        let comp_rect = egui::Rect::from_center_size(comp.position * zoom, comp.size * zoom);
        max_y = max_y.max(comp_rect.bottom());
    }

    let route_y = max_y + clearance;

    vec![
        egui::pos2(from.center().x, from.top() - edge_clearance),
        egui::pos2(from.center().x, route_y),
        egui::pos2(to.center().x, route_y),
        egui::pos2(to.center().x, to.bottom() + edge_clearance),
    ]
}

// Create a bypass path that avoids obstacles
fn create_bypass_path(
    start: Pos2,
    end: Pos2,
    from_rect: Rect,
    to_rect: Rect,
    components: &[DraggableComponent],
    zoom: f32,
) -> Vec<Pos2> {
    let clearance = 40.0 * zoom;
    let dx = end.x - start.x;
    let dy = end.y - start.y;

    if dx.abs() > dy.abs() {
        // Horizontal primary direction - route above or below
        let route_above_y = from_rect.top().min(to_rect.top()) - clearance;
        let route_below_y = from_rect.bottom().max(to_rect.bottom()) + clearance;

        // Check which route has fewer obstacles
        let above_path = vec![
            start,
            egui::pos2(start.x, route_above_y),
            egui::pos2(end.x, route_above_y),
            end,
        ];

        if count_intersections(&above_path, components, zoom) == 0 {
            return above_path;
        }

        // Route below
        vec![
            start,
            egui::pos2(start.x, route_below_y),
            egui::pos2(end.x, route_below_y),
            end,
        ]
    } else {
        // Vertical primary direction - route left or right
        let route_left_x = from_rect.left().min(to_rect.left()) - clearance;
        let route_right_x = from_rect.right().max(to_rect.right()) + clearance;

        // Check which route has fewer obstacles
        let left_path = vec![
            start,
            egui::pos2(route_left_x, start.y),
            egui::pos2(route_left_x, end.y),
            end,
        ];

        if count_intersections(&left_path, components, zoom) == 0 {
            return left_path;
        }

        // Route right
        vec![
            start,
            egui::pos2(route_right_x, start.y),
            egui::pos2(route_right_x, end.y),
            end,
        ]
    }
}

// Check if a path is clear of obstacles
fn is_path_clear(path: &[Pos2], from_id: &str, to_id: &str, components: &[DraggableComponent], zoom: f32) -> bool {
    for i in 0..path.len() - 1 {
        let segment_start = path[i];
        let segment_end = path[i + 1];

        for comp in components {
            // Skip source and target components
            if comp.id == from_id || comp.id == to_id {
                continue;
            }

            let comp_rect =
                egui::Rect::from_center_size(comp.position * zoom, comp.size * zoom)
                    .expand(25.0 * zoom); // Increased clearance, scaled by zoom

            if line_intersects_rect(segment_start, segment_end, comp_rect) {
                return false;
            }
        }
    }
    true
}

// Count intersections for path quality evaluation
fn count_intersections(path: &[Pos2], components: &[DraggableComponent], zoom: f32) -> usize {
    let mut count = 0;
    for i in 0..path.len() - 1 {
        for comp in components {
            let comp_rect =
                egui::Rect::from_center_size(comp.position * zoom, comp.size * zoom)
                    .expand(25.0 * zoom); // Increased clearance, scaled by zoom

            if line_intersects_rect(path[i], path[i + 1], comp_rect) {
                count += 1;
            }
        }
    }
    count
}

// Simple line-rectangle intersection test
fn line_intersects_rect(p1: Pos2, p2: Pos2, rect: Rect) -> bool {
    // Check if either endpoint is inside the rectangle
    if rect.contains(p1) || rect.contains(p2) {
        return true;
    }

    // Check if line is completely outside rectangle bounds
    if (p1.x < rect.left() && p2.x < rect.left())
        || (p1.x > rect.right() && p2.x > rect.right())
        || (p1.y < rect.top() && p2.y < rect.top())
        || (p1.y > rect.bottom() && p2.y > rect.bottom())
    {
        return false;
    }

    // Check intersection with each edge
    let edges = [
        (rect.left_top(), rect.right_top()),       // Top
        (rect.right_top(), rect.right_bottom()),   // Right
        (rect.right_bottom(), rect.left_bottom()), // Bottom
        (rect.left_bottom(), rect.left_top()),     // Left
    ];

    for (edge_start, edge_end) in edges {
        if lines_intersect(p1, p2, edge_start, edge_end) {
            return true;
        }
    }

    false
}

// Check if two line segments intersect
fn lines_intersect(p1: Pos2, p2: Pos2, p3: Pos2, p4: Pos2) -> bool {
    let d1 = (p4.y - p3.y) * (p2.x - p1.x) - (p4.x - p3.x) * (p2.y - p1.y);
    if d1.abs() < 0.001 {
        return false; // Lines are parallel
    }

    let ua = ((p4.x - p3.x) * (p1.y - p3.y) - (p4.y - p3.y) * (p1.x - p3.x)) / d1;
    let ub = ((p2.x - p1.x) * (p1.y - p3.y) - (p2.y - p1.y) * (p1.x - p3.x)) / d1;

    ua >= 0.0 && ua <= 1.0 && ub >= 0.0 && ub <= 1.0
}
