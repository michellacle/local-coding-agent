#!/usr/bin/env python3
"""
Simple To-Do List Manager Application
Demonstrates basic Python program structure
"""

def add_task(tasks, task_name):
    """Add a new task to the list"""
    tasks.append(task_name)
    return True

def complete_task(tasks, index):
    """Mark a task as completed (removes it)"""
    if 0 <= index < len(tasks):
        del tasks[index]
        return True
    return False

def display_tasks(tasks):
    """Display all current tasks in a readable format"""
    print("\nTo-Do List:")
    print("=" * 50)
    if not tasks:
        print("No tasks to display.")
    else:
        for i, task in enumerate(tasks):
            print(f"{i+1}. {task}")
    print("=" * 50)
    return tasks

def main():
    """Main function - the entry point of the program"""
    # Initialize tasks list
    todo_list = []
    
    # Welcome message
    print("\nWelcome to Python To-Do List Manager!")
    print(f"Running: {__file__}")
    print("-" * 40)
    
    # Add some sample tasks
    print("Sample Tasks Being Added:")
    add_task(todo_list, "Complete your first Python program")
    print("✓ Task added!")
    add_task(todo_list, "Learn to use automated tools")
    print("✓ Task added!")
    add_task(todo_list, "Build functional scripts")
    print("✓ Task added!")
    
    # Display initial list
    display_tasks(todo_list)
    
    # Remove the first task
    if todo_list:
        index_to_remove = 0
        complete_task(todo_list, index_to_remove)
        print(f"\nTask #{index_to_remove + 1} removed!")
    
    display_tasks(todo_list)

if __name__ == "__main__":
    main()