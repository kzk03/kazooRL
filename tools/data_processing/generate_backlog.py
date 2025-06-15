import json
import random


def generate_backlog(num_tasks=20, output_path="data/backlog.json"):
    backlog = []
    all_skills = ["python", "java", "c++", "javascript", "go", "rust"]

    for i in range(num_tasks):
        task = {
            "id": f"TASK-{i}",
            "name": f"Task number {i}",
            "required_skills": random.sample(all_skills, k=random.randint(1, 2)),
            "complexity": random.randint(1, 10),
        }
        backlog.append(task)

    with open(output_path, "w") as f:
        json.dump(backlog, f, indent=4)

    print(f"Generated {num_tasks} tasks in {output_path}")


if __name__ == "__main__":
    generate_backlog()
