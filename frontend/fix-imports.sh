#!/bin/bash

# Fix imports in all TypeScript/React files
echo "Fixing imports in TypeScript files..."

# Function to fix imports based on file location
fix_imports() {
    local file=$1
    local depth=$(echo "$file" | tr -cd '/' | wc -c)

    # Calculate relative path prefix based on depth
    local prefix=""
    if [ $depth -eq 2 ]; then
        prefix="../"
    elif [ $depth -eq 3 ]; then
        prefix="../../"
    elif [ $depth -eq 4 ]; then
        prefix="../../../"
    fi

    # Fix imports in the file
    sed -i.bak \
        -e "s|from '@/types'|from '${prefix}types'|g" \
        -e "s|from '@/hooks/|from '${prefix}hooks/|g" \
        -e "s|from '@/utils/|from '${prefix}utils/|g" \
        -e "s|from '@/context/|from '${prefix}context/|g" \
        -e "s|from '@/components/|from '${prefix}components/|g" \
        -e "s|from '@/services/|from '${prefix}services/|g" \
        "$file"
}

# Process all TypeScript files
for file in $(find src -name "*.tsx" -o -name "*.ts"); do
    if grep -q "@/" "$file"; then
        echo "Fixing: $file"
        fix_imports "$file"
    fi
done

# Clean up backup files
find src -name "*.bak" -delete

echo "Import fixes complete!"