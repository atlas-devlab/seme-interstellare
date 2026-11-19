"""Small bounded interpreter for a declared stack32 message format.

The receiver knows JSON, uint32, a stack, and the primitive operations below.
The message supplies the opcode mapping, examples, program and initial values.
This is an algorithm demonstrator, not a transported AI model.
"""
from __future__ import annotations

import json
import struct

WORD_MASK = (1 << 32)-1
MAX_CAPSULE_BYTES = 256*1024
MAX_CODE_BYTES = 16384
MAX_STEPS = 20000
MAX_STACK = 512
MAX_OUTPUT = 1024
MAX_TESTS = 32
ARITY = {name: 0 for name in ['HALT', 'ADD', 'SUB', 'MUL', 'DUP', 'SWAP',
                             'OVER', 'ROT', 'DROP', 'OUT']}
ARITY.update(PUSH=1, JMP=1, JZ=1)
DESCRIPTIONS = {
    'HALT': 'Stop and return the output produced so far.',
    'PUSH': 'Push the following uint32 operand on the stack.',
    'ADD': 'Pop b,a; push (a+b) modulo 2^32.',
    'SUB': 'Pop b,a; push (a-b) modulo 2^32.',
    'MUL': 'Pop b,a; push (a*b) modulo 2^32.',
    'DUP': 'a -> a,a', 'SWAP': 'a,b -> b,a',
    'OVER': 'a,b -> a,b,a', 'ROT': 'a,b,c -> b,c,a',
    'DROP': 'Remove the top stack item.', 'OUT': 'Pop one item and append it to output.',
    'JMP': 'Jump to the zero-based instruction index in the uint32 operand.',
    'JZ': 'Pop a; jump to the operand instruction index if a equals zero.',
}


class VMError(ValueError):
    pass


def integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise VMError(f'{label}: expected integer in [{low}, {high}]')
    return value


def default_machine():
    order = ['HALT', 'PUSH', 'ADD', 'SUB', 'MUL', 'DUP', 'SWAP', 'OVER',
             'ROT', 'DROP', 'OUT', 'JMP', 'JZ']
    return dict(name='SIF-stack32', word_bits=32, arithmetic='unsigned_modulo_2pow32',
                operand_encoding='uint32_little_endian', jump_unit='instruction_index_zero_based',
                limits=dict(steps=10000, stack=256, output=512),
                opcode_table=[dict(opcode=i, operation=name, operands=ARITY[name],
                                   definition=DESCRIPTIONS[name]) for i,name in enumerate(order)])


def machine_contract(machine):
    if type(machine) is not dict:
        raise VMError('Machine specification must be an object')
    if (type(machine.get('word_bits')) is not int or machine['word_bits'] != 32
            or machine.get('arithmetic') != 'unsigned_modulo_2pow32'
            or machine.get('operand_encoding') != 'uint32_little_endian'
            or machine.get('jump_unit') != 'instruction_index_zero_based'):
        raise VMError('Unsupported machine arithmetic or operand convention')
    rows = machine.get('opcode_table')
    if type(rows) is not list or not 1 <= len(rows) <= len(ARITY):
        raise VMError('Invalid opcode table size')
    table, names = {}, set()
    for row in rows:
        if type(row) is not dict:
            raise VMError('Invalid opcode row')
        opcode = integer(row.get('opcode'), 0, 255, 'opcode')
        operation = row.get('operation')
        if not isinstance(operation, str) or operation not in ARITY:
            raise VMError('Unknown primitive operation')
        if opcode in table or operation in names:
            raise VMError('Duplicate opcode or operation')
        if type(row.get('operands')) is not int or row['operands'] != ARITY[operation]:
            raise VMError('Operand count disagrees with host primitive')
        table[opcode] = operation
        names.add(operation)
    raw_limits = machine.get('limits')
    if type(raw_limits) is not dict:
        raise VMError('Missing limits')
    limits = dict(steps=integer(raw_limits.get('steps'), 1, MAX_STEPS, 'steps'),
                  stack=integer(raw_limits.get('stack'), 1, MAX_STACK, 'stack'),
                  output=integer(raw_limits.get('output'), 1, MAX_OUTPUT, 'output'))
    return table, limits


def assemble(machine, instructions):
    table, _ = machine_contract(machine)
    codes = {name: code for code,name in table.items()}
    output = bytearray()
    for instruction in instructions:
        if type(instruction) not in [list, tuple] or not instruction:
            raise VMError('Instruction must be a nonempty sequence')
        name, *operands = instruction
        if name not in codes or len(operands) != ARITY[name]:
            raise VMError('Unknown instruction or wrong operand count')
        output.append(codes[name])
        for operand in operands:
            output.extend(struct.pack('<I', integer(operand,0,WORD_MASK,'operand')))
    if len(output) > MAX_CODE_BYTES:
        raise VMError('Program exceeds byte limit')
    return bytes(output).hex()


def execute(machine, bytecode_hex):
    table, limits = machine_contract(machine)
    if (not isinstance(bytecode_hex,str) or len(bytecode_hex) > 2*MAX_CODE_BYTES
            or len(bytecode_hex)%2 or any(c not in '0123456789abcdef' for c in bytecode_hex)):
        raise VMError('Invalid bytecode hex or program size')
    code = bytes.fromhex(bytecode_hex)
    instructions, position = [], 0
    while position < len(code):
        opcode = code[position]
        position += 1
        if opcode not in table:
            raise VMError('Bytecode uses an undeclared opcode')
        name = table[opcode]
        operand = None
        if ARITY[name]:
            if position+4 > len(code):
                raise VMError('Truncated operand')
            operand = struct.unpack_from('<I',code,position)[0]
            position += 4
        instructions.append((name,operand))
    if not any(name=='HALT' for name,_ in instructions):
        raise VMError('Program contains no HALT')
    for name,operand in instructions:
        if name in ['JMP','JZ'] and operand >= len(instructions):
            raise VMError('Jump outside instruction array')
    stack, output, pc = [], [], 0
    required = dict(ADD=2,SUB=2,MUL=2,DUP=1,SWAP=2,OVER=2,ROT=3,DROP=1,OUT=1,JZ=1)
    for step in range(1,limits['steps']+1):
        if not 0 <= pc < len(instructions):
            raise VMError('Execution fell outside the program before HALT')
        name,operand = instructions[pc]
        pc += 1
        if len(stack) < required.get(name,0):
            raise VMError('Stack underflow')
        if name=='HALT':
            return dict(output=output,steps=step,final_stack=stack,halted=True)
        if name=='PUSH':
            stack.append(operand)
        elif name in ['ADD','SUB','MUL']:
            b,a = stack.pop(),stack.pop()
            value = a+b if name=='ADD' else a-b if name=='SUB' else a*b
            stack.append(value & WORD_MASK)
        elif name=='DUP':
            stack.append(stack[-1])
        elif name=='SWAP':
            stack[-2:] = [stack[-1],stack[-2]]
        elif name=='OVER':
            stack.append(stack[-2])
        elif name=='ROT':
            a,b,c = stack[-3:]
            stack[-3:] = [b,c,a]
        elif name=='DROP':
            stack.pop()
        elif name=='OUT':
            output.append(stack.pop())
        elif name=='JMP':
            pc = operand
        elif name=='JZ':
            if stack.pop()==0:
                pc = operand
        if len(stack)>limits['stack'] or len(output)>limits['output']:
            raise VMError('Stack or output resource limit exceeded')
    raise VMError('Instruction step limit exceeded')


def no_duplicate_keys(pairs):
    result = {}
    for key,value in pairs:
        if key in result:
            raise VMError('Duplicate JSON key')
        result[key] = value
    return result


def parse_integer(token):
    if len(token.lstrip('-')) > 20:
        raise VMError('Malformed capsule JSON: integer exceeds the format limit')
    return int(token)


def reject_constant(token):
    raise VMError('Malformed capsule JSON: non-finite number')


def read_json(payload):
    if type(payload) is not bytes or len(payload)>MAX_CAPSULE_BYTES:
        raise VMError('Capsule size or type invalid')
    try:
        return json.loads(payload.decode('utf-8'), object_pairs_hook=no_duplicate_keys,
                          parse_int=parse_integer, parse_constant=reject_constant)
    except VMError:
        raise
    except (UnicodeError,ValueError,RecursionError) as error:
        raise VMError('Malformed capsule JSON') from error


def receive_capsule(payload):
    capsule = read_json(payload)
    if type(capsule) is not dict or capsule.get('format') != 'SIF-SEED/1':
        raise VMError('Unsupported capsule format')
    machine = capsule.get('machine')
    machine_contract(machine)
    tests = capsule.get('test_vectors')
    if type(tests) is not list or not 1 <= len(tests)<=MAX_TESTS:
        raise VMError('Missing or excessive machine test vectors')
    for test in tests:
        if type(test) is not dict:
            raise VMError('Invalid test vector')
        expected = test.get('expected_output')
        if type(expected) is not list or len(expected)>MAX_OUTPUT:
            raise VMError('Invalid expected output')
        for value in expected:
            integer(value,0,WORD_MASK,'expected value')
        if execute(machine,test.get('bytecode_hex'))['output'] != expected:
            raise VMError('Machine test vector failed')
    program = capsule.get('program')
    if type(program) is not dict:
        raise VMError('Missing program')
    result = execute(machine,program.get('bytecode_hex'))
    result.update(program_name=program.get('name'),machine_tests_passed=len(tests),
                  contains_AI_model=False,original_Wow_content=False)
    return result


def make_seed(first=0, second=1, terms=16, machine=None):
    integer(first,0,WORD_MASK,'first seed')
    integer(second,0,WORD_MASK,'second seed')
    integer(terms,1,256,'term count')
    machine = default_machine() if machine is None else machine
    tests = []
    for name,instructions,answer in [
        ('addition',[('PUSH',1),('PUSH',2),('ADD',),('OUT',),('HALT',)],[3]),
        ('unsigned_wrap',[('PUSH',WORD_MASK),('PUSH',1),('ADD',),('OUT',),('HALT',)],[0]),
        ('subtraction',[('PUSH',0),('PUSH',1),('SUB',),('OUT',),('HALT',)],[WORD_MASK]),
        ('multiplication',[('PUSH',6),('PUSH',7),('MUL',),('OUT',),('HALT',)],[42]),
        ('stack_rules',[('PUSH',2),('PUSH',3),('OVER',),('ADD',),('SWAP',),('OUT',),('OUT',),('HALT',)],[2,5]),
        ('conditional',[('PUSH',0),('JZ',4),('PUSH',999),('OUT',),('PUSH',7),('OUT',),('HALT',)],[7]),
    ]:
        tests.append(dict(name=name,bytecode_hex=assemble(machine,instructions),expected_output=answer))
    instructions = [('PUSH',first),('PUSH',second)]
    for _ in range(terms):
        instructions.extend([('OVER',),('OUT',),('DUP',),('ROT',),('ADD',)])
    instructions.append(('HALT',))
    capsule = dict(format='SIF-SEED/1',kind='constructed_algorithm_demonstrator',
        introduction=dict(
            host_assumptions=['UTF-8/JSON','uint32 arithmetic','stack semantics','listed primitive operation names','SIF1 framing'],
            teaching='The message carries opcode mapping, operation definitions and worked test vectors.',
            limit='It does not teach an unknown civilization from no shared conventions; it requires the declared receiver.'),
        machine=machine,test_vectors=tests,
        program=dict(name='fibonacci_mod_2pow32',initial_seeds=[first,second],terms=terms,
                     bytecode_hex=assemble(machine,instructions)),
        provenance=dict(authored_demo=True,decoded_from_Wow=False,contains_AI_model=False))
    return json.dumps(capsule,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii')
