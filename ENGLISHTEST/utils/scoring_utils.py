class AnswerScoring:
    def __init__(self):
        self.question_weights = {}  
        self.user_answers = {} 
        self.correct_answers = {}
        self.question_count = 0 

    def add_question(self, question_id, correct_answer, weight=1.0):
        self.question_weights[question_id] = weight
        self.correct_answers[question_id] = correct_answer.upper()
        self.question_count += 1

    def record_user_answer(self, question_id, user_answer):
        self.user_answers[question_id] = user_answer.upper()

    def calculate_score(self):
        if not self.user_answers:
            return 0, 0, []

        total_weight = sum(self.question_weights.values())
        total_score = 0
        wrong_questions = []

        for question_id in self.correct_answers:
            if question_id in self.user_answers:
                user_ans = self.user_answers[question_id]
                correct_ans = self.correct_answers[question_id]
                weight = self.question_weights[question_id]

                if user_ans == correct_ans:
                    total_score += weight
                else:
                    wrong_questions.append({
                        'question_id': question_id,
                        'user_answer': user_ans,
                        'correct_answer': correct_ans
                    })

        percentage_score = (total_score / total_weight) * 100 if total_weight > 0 else 0
        return total_score, percentage_score, wrong_questions

    def get_statistics(self):
        total_questions = len(self.correct_answers)
        answered_questions = len(self.user_answers)
        correct_count = sum(1 for q_id in self.user_answers 
                          if self.user_answers[q_id] == self.correct_answers.get(q_id))

        return {
            'total_questions': total_questions,
            'answered_questions': answered_questions,
            'correct_count': correct_count,
            'accuracy': (correct_count / answered_questions * 100) if answered_questions > 0 else 0
        } 