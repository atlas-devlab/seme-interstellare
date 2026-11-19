"""Independent behavioral checks for the bounded stack32 receiver."""
import json
import unittest

import macchina as vm


def reference_sequence(first, second, terms):
    values = []
    for _ in range(terms):
        values.append(first)
        first, second = second, (first + second) & vm.WORD_MASK
    return values


class MachineTests(unittest.TestCase):
    def run_code(self, instructions, machine=None):
        machine = machine or vm.default_machine()
        return vm.execute(machine, vm.assemble(machine, instructions))

    def test_fibonacci_independent_seeds(self):
        for first, second, terms in [(0, 1, 18), (2, 3, 16), (9, 11, 1),
                                      (vm.WORD_MASK, vm.WORD_MASK, 20)]:
            with self.subTest(seeds=(first, second), terms=terms):
                result = vm.receive_capsule(vm.make_seed(first, second, terms))
                self.assertEqual(result['output'], reference_sequence(first, second, terms))
                self.assertTrue(result['halted'])
                self.assertFalse(result['contains_AI_model'])

    def test_transmitted_opcode_mapping_and_row_order(self):
        machine = vm.default_machine()
        machine['opcode_table'].reverse()
        for index, row in enumerate(machine['opcode_table']):
            row['opcode'] = 200 + index
        result = vm.receive_capsule(vm.make_seed(2, 3, 12, machine))
        self.assertEqual(result['output'], reference_sequence(2, 3, 12))

    def test_unsigned_arithmetic_edges(self):
        for operation, left, right, expected in [
            ('ADD', vm.WORD_MASK, 1, 0),
            ('SUB', 0, 1, vm.WORD_MASK),
            ('MUL', 65536, 65536, 0),
            ('MUL', vm.WORD_MASK, 2, vm.WORD_MASK - 1),
        ]:
            with self.subTest(operation=operation):
                result = self.run_code([('PUSH', left), ('PUSH', right),
                                        (operation,), ('OUT',), ('HALT',)])
                self.assertEqual(result['output'], [expected])

    def test_uint32_little_endian_operand(self):
        machine = vm.default_machine()
        encoded = vm.assemble(machine, [('PUSH', 0x12345678), ('HALT',)])
        self.assertEqual(bytes.fromhex(encoded)[1:5], b'\x78\x56\x34\x12')

    def test_stack_operations(self):
        result = self.run_code([('PUSH', 1), ('PUSH', 2), ('PUSH', 3),
                                ('ROT',), ('DUP',), ('DROP',), ('OVER',),
                                ('SWAP',), ('OUT',), ('OUT',), ('OUT',), ('OUT',), ('HALT',)])
        self.assertEqual(result['output'], [1, 3, 3, 2])

    def test_unknown_primitive(self):
        machine = vm.default_machine()
        machine['opcode_table'][0]['operation'] = 'SHELL'
        with self.assertRaises(vm.VMError):
            vm.machine_contract(machine)

    def test_duplicate_opcode_and_operation(self):
        machine = vm.default_machine()
        machine['opcode_table'][1]['opcode'] = machine['opcode_table'][0]['opcode']
        with self.assertRaises(vm.VMError):
            vm.machine_contract(machine)
        machine = vm.default_machine()
        machine['opcode_table'][1].update(operation='HALT', operands=0)
        with self.assertRaises(vm.VMError):
            vm.machine_contract(machine)

    def test_wrong_arity_and_bool_fields(self):
        for field, value in [('operands', 1), ('operands', False), ('opcode', True)]:
            machine = vm.default_machine()
            machine['opcode_table'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(vm.VMError):
                vm.machine_contract(machine)
        with self.assertRaises(vm.VMError):
            vm.assemble(vm.default_machine(), [('PUSH', True), ('HALT',)])

    def test_unsupported_convention(self):
        for field, value in [('word_bits', 64), ('word_bits', True),
                             ('arithmetic', 'signed'), ('operand_encoding', 'big_endian'),
                             ('jump_unit', 'byte_offset')]:
            machine = vm.default_machine()
            machine[field] = value
            with self.subTest(field=field), self.assertRaises(vm.VMError):
                vm.machine_contract(machine)

    def test_truncated_operand_and_invalid_hex(self):
        machine = vm.default_machine()
        for encoded in ['01', '010000', '0', 'zz', 'FF', '00ff']:
            with self.subTest(encoded=encoded), self.assertRaises(vm.VMError):
                vm.execute(machine, encoded)

    def test_unknown_assembly_instruction_and_arity(self):
        for instructions in [[('NOOP',)], [('PUSH',)], [('ADD', 1)], [('PUSH', -1)]]:
            with self.subTest(instructions=instructions), self.assertRaises(vm.VMError):
                vm.assemble(vm.default_machine(), instructions)

    def test_jumps_use_instruction_indices(self):
        result = self.run_code([('JMP', 3), ('PUSH', 99), ('OUT',),
                                ('PUSH', 7), ('OUT',), ('HALT',)])
        self.assertEqual(result['output'], [7])
        for condition, expected in [(0, [7]), (1, [99, 7])]:
            result = self.run_code([('PUSH', condition), ('JZ', 4), ('PUSH', 99),
                                    ('OUT',), ('PUSH', 7), ('OUT',), ('HALT',)])
            self.assertEqual(result['output'], expected)

    def test_out_of_range_jump_even_if_unreachable(self):
        with self.assertRaises(vm.VMError):
            self.run_code([('HALT',), ('JMP', 2)])

    def test_stack_underflow(self):
        for operation in ['ADD', 'SUB', 'MUL', 'DUP', 'SWAP', 'OVER', 'ROT',
                          'DROP', 'OUT']:
            with self.subTest(operation=operation), self.assertRaises(vm.VMError):
                self.run_code([(operation,), ('HALT',)])
        with self.assertRaises(vm.VMError):
            self.run_code([('JZ', 1), ('HALT',)])

    def test_missing_halt_and_falling_off_program(self):
        with self.assertRaises(vm.VMError):
            self.run_code([('PUSH', 1)])
        with self.assertRaises(vm.VMError):
            self.run_code([('JMP', 2), ('HALT',), ('PUSH', 1)])

    def test_infinite_loop_hits_step_limit(self):
        machine = vm.default_machine()
        machine['limits']['steps'] = 5
        with self.assertRaisesRegex(vm.VMError, 'step limit'):
            self.run_code([('JMP', 0), ('HALT',)], machine)

    def test_stack_and_output_limits(self):
        machine = vm.default_machine()
        machine['limits']['stack'] = 1
        with self.assertRaisesRegex(vm.VMError, 'resource limit'):
            self.run_code([('PUSH', 1), ('DUP',), ('HALT',)], machine)
        machine = vm.default_machine()
        machine['limits']['output'] = 1
        with self.assertRaisesRegex(vm.VMError, 'resource limit'):
            self.run_code([('PUSH', 1), ('OUT',), ('PUSH', 2), ('OUT',), ('HALT',)], machine)

    def test_declared_limits_cannot_exceed_host_limits(self):
        for name, value in [('steps', vm.MAX_STEPS + 1), ('stack', vm.MAX_STACK + 1),
                            ('output', vm.MAX_OUTPUT + 1), ('steps', 0), ('output', True)]:
            machine = vm.default_machine()
            machine['limits'][name] = value
            with self.subTest(name=name), self.assertRaises(vm.VMError):
                vm.machine_contract(machine)

    def test_altered_test_vector_rejected(self):
        capsule = json.loads(vm.make_seed())
        capsule['test_vectors'][0]['expected_output'] = [999]
        with self.assertRaisesRegex(vm.VMError, 'test vector failed'):
            vm.receive_capsule(json.dumps(capsule).encode('utf-8'))

    def test_invalid_expected_output_and_test_counts(self):
        for expected in [[True], [-1], [vm.WORD_MASK + 1], '3']:
            capsule = json.loads(vm.make_seed())
            capsule['test_vectors'][0]['expected_output'] = expected
            with self.subTest(expected=expected), self.assertRaises(vm.VMError):
                vm.receive_capsule(json.dumps(capsule).encode())
        for tests in [[], [None], [{}] * (vm.MAX_TESTS + 1)]:
            capsule = json.loads(vm.make_seed())
            capsule['test_vectors'] = tests
            with self.subTest(tests_count=len(tests)), self.assertRaises(vm.VMError):
                vm.receive_capsule(json.dumps(capsule).encode())

    def test_duplicate_json_top_level_and_nested(self):
        for payload in [b'{"format":"SIF-SEED/1","format":"SIF-SEED/1"}',
                        b'{"a":{"x":1,"x":2}}']:
            with self.subTest(payload=payload), self.assertRaisesRegex(vm.VMError, 'Duplicate JSON key'):
                vm.receive_capsule(payload)

    def test_capsule_and_program_size_limits(self):
        with self.assertRaises(vm.VMError):
            vm.read_json(b' ' * (vm.MAX_CAPSULE_BYTES + 1))
        with self.assertRaises(vm.VMError):
            vm.execute(vm.default_machine(), '00' * (vm.MAX_CODE_BYTES + 1))

    def test_malformed_capsules(self):
        for payload in [b'\xff', b'{', b'[]', b'null', b'{"format":"other"}']:
            with self.subTest(payload=payload), self.assertRaises(vm.VMError):
                vm.receive_capsule(payload)

    def test_oversized_json_integer_is_controlled_error(self):
        with self.assertRaises(vm.VMError):
            vm.read_json(b'{"x":' + b'9' * 5000 + b'}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
