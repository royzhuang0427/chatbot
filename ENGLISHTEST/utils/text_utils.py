def save_source_in_txt(texts, output_txt_path):
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(texts, 1):
                f.write(f"--- PDF {i} ---\n")
                f.write(text)
                f.write("\n\n")
        print(f"已保存至 {output_txt_path}")
    except Exception as e:
        print(f"保存時出錯: {str(e)}") 