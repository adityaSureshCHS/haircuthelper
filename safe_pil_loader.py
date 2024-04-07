def safe_pil_loader(path):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except (OSError, SyntaxError):
            return Image.new('RGB', (224, 224))