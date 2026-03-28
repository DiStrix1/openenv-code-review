from env.models import PullRequest, Issue
class BaseTask:
    name = "base"
    complexity = "medium"
    objective = "Review code and identify issues."
    developer_profile = {
        "fix_multiplier": 1.0,
        "feedback_boost": 0.2,
        "unprompted_factor": 0.5,
        "max_fix_prob": 0.6,
        "bug_introduce_prob": 0.3,
        "bug_severity_weights": [0.5, 0.3, 0.2],
    }

    def create_pr(self):
        raise NotImplementedError()

    def get_developer_profile(self):
        return self.developer_profile

class EasyTask(BaseTask):
    name = "easy"
    complexity = "easy"
    objective = "Catch obvious correctness and safety issues in a short diff."
    developer_profile = {
        "fix_multiplier": 1.2,
        "feedback_boost": 0.25,
        "unprompted_factor": 0.6,
        "max_fix_prob": 0.75,
        "bug_introduce_prob": 0.15,
        "bug_severity_weights": [0.65, 0.25, 0.10],
    }

    def create_pr(self):
        issues =[
            Issue(id = 1, description="Assignment instead of comparison", severity = "high"),
            Issue(id = 2, description="Missing null check", severity = "medium"),
        ]
        diff = """
        if(x = 5){
            process(user.name);
        }"""
        return PullRequest(diff = diff, issues=issues)

class MediumTask(BaseTask):
    name = "medium"
    complexity = "medium"
    objective = "Identify performance and safety issues in a noisier change."
    developer_profile = {
        "fix_multiplier": 1.0,
        "feedback_boost": 0.2,
        "unprompted_factor": 0.5,
        "max_fix_prob": 0.6,
        "bug_introduce_prob": 0.3,
        "bug_severity_weights": [0.5, 0.3, 0.2],
    }

    def create_pr(self):
        issues = [
            Issue(id=1, description="Inefficient loop", severity="low"),
            Issue(id=2, description="Possible null pointer", severity="high"),
        ]
        diff = """
        for(int i=0;i<list.size();i++){
            for(int j=0;j<list.size();j++){
                print(list.get(j).name);
            }
        }
        """
        return PullRequest(diff=diff, issues=issues)

class HardTask(BaseTask):
    name = "hard"
    complexity = "hard"
    objective = "Balance correctness, performance, and safety across interacting defects."
    developer_profile = {
        "fix_multiplier": 0.85,
        "feedback_boost": 0.15,
        "unprompted_factor": 0.35,
        "max_fix_prob": 0.5,
        "bug_introduce_prob": 0.45,
        "bug_severity_weights": [0.35, 0.35, 0.30],
    }

    def create_pr(self):
        issues = [
            Issue(id=1, description="Race condition in shared counter", severity="high"),
            Issue(id=2, description="Memory leak due to object retention", severity="high"),
            Issue(id=3, description="Nested loop inefficiency", severity="medium"),
            Issue(id=4, description="Unused variable", severity="low"),
            Issue(id=5, description="Assignment in condition", severity="high")
        ]

        diff = """
        global_counter++;

        for(int i=0;i<list.size();i++){
                process(list.get(i));
        }

        cache.add(new Object());

        int temp;

        if (x = 5) {
            doSomething();
        }
        """

        return PullRequest(diff=diff, issues=issues)