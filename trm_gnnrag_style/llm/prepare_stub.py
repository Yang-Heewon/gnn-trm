import json


def build_prompt(question: str, relation_path):
    rel_seq = ' -> '.join(str(x) for x in relation_path)
    return f'Question: {question}\nCandidate relation path: {rel_seq}\nAnswer:'


def main():
    # lightweight stub for future extension
    sample = {'question': 'Who wrote Dune?', 'path': ['book.author']}
    print(json.dumps({'prompt': build_prompt(sample['question'], sample['path'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
