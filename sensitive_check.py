class SensitiveWordError(Exception):
    pass


class DFA:
    def __init__(self):
        self.start_state = 0
        self.accept_states = set()
        self.transitions = {}

    def add_word(self, word):
        current_state = self.start_state
        for char in word:
            next_state = self.transitions.get((current_state, char))
            if next_state is None:
                next_state = len(self.transitions)
                self.transitions[(current_state, char)] = next_state
            current_state = next_state
        self.accept_states.add(current_state)

    def is_match(self, text):
        current_state = self.start_state
        for char in text:
            next_state = self.transitions.get((current_state, char))
            if next_state is None:
                current_state = self.start_state
            else:
                current_state = next_state
            if current_state in self.accept_states:
                return True
        return False


def build_dfa(words_file_path):
    dfa = DFA()
    with open(words_file_path, 'r') as f:
        for line in f:
            word = line.strip()
            dfa.add_word(word)
    return dfa

