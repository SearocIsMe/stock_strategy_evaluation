#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复验证脚本
验证 AttributeError: '_CurrentObj' object has no attribute 'high' 修复
"""

import sys
import os
import re

def test_fix():
    """测试修复是否成功"""
    print("🔍 测试 AttributeError 修复...")
    
    # 读取修复后的文件
    file_path = "joinquant/六一中路_优化版.py"
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否还有 .high 的错误用法（排除 .high_limit）
    high_pattern = r'\.high(?!\_limit)'
    problematic_matches = re.findall(high_pattern, content)
    
    if problematic_matches:
        print(f"❌ 仍然发现 {len(problematic_matches)} 个问题:")
        for match in problematic_matches:
            print(f"   - {match}")
        return False
    
    # 检查修复后的代码是否存在
    if 'attribute_history(stock, 1, \'1d\', [\'high\', \'close\'])' in content:
        print("✅ 找到修复后的代码，使用 attribute_history 获取历史数据")
    else:
        print("⚠️  未找到预期的修复代码")
    
    # 检查异常处理
    if 'except Exception as e:' in content and 'log.warn' in content:
        print("✅ 找到异常处理代码")
    
    # 检查是否还有 current_data[stock].high 的直接访问
    if 'current_data[stock].high' in content:
        print("❌ 仍然存在 current_data[stock].high 的直接访问")
        return False
    
    print("✅ AttributeError 修复验证通过!")
    return True

def test_syntax():
    """测试语法是否正确"""
    print("\n🔍 测试文件语法...")
    
    try:
        import ast
        file_path = "joinquant/六一中路_优化版.py"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 尝试解析 AST
        ast.parse(content)
        print("✅ 文件语法正确")
        return True
        
    except SyntaxError as e:
        print(f"❌ 语法错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 开始验证 AttributeError 修复")
    print("=" * 60)
    
    fix_ok = test_fix()
    syntax_ok = test_syntax()
    
    print("\n" + "=" * 60)
    if fix_ok and syntax_ok:
        print("🎉 所有测试通过! 修复成功!")
        sys.exit(0)
    else:
        print("❌ 测试失败，需要进一步检查")
        sys.exit(1)