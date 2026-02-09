/**
 * Kore Edge — Node.js WASM inference example.
 *
 * Build: wasm-pack build crates/kore-edge --target nodejs --release
 * Run:   node crates/kore-edge/examples/wasm-node/main.js model.koref
 */

const fs = require('fs');
const path = require('path');

async function main() {
    const modelPath = process.argv[2];
    if (!modelPath) {
        console.error('Usage: node main.js <model.koref>');
        process.exit(1);
    }

    // Import the WASM module (built with wasm-pack --target nodejs)
    const kore = require('../../pkg/kore_edge.js');

    console.log('Loading model:', modelPath);
    const bytes = fs.readFileSync(modelPath);
    const session = kore.KoreSession.fromBytes(new Uint8Array(bytes));
    console.log('Model info:', session.info());

    // Forward pass
    const inputIds = new Uint32Array([1, 2, 3, 4, 5]);
    console.log('\nInput IDs:', Array.from(inputIds));

    const start = performance.now();
    const logits = session.forward(inputIds);
    const forwardMs = performance.now() - start;
    console.log(`Forward pass: ${forwardMs.toFixed(1)}ms`);
    console.log(`Logits shape: [${logits.length}]`);

    // Top-5 logits
    const indexed = Array.from(logits).map((v, i) => ({ v, i }));
    indexed.sort((a, b) => b.v - a.v);
    console.log('Top 5 logits:');
    for (const { v, i } of indexed.slice(0, 5)) {
        console.log(`  [${i}] = ${v.toFixed(4)}`);
    }

    // Generation
    console.log('\nGenerating 16 tokens...');
    const genStart = performance.now();
    const tokens = session.generate(inputIds, 16);
    const genMs = performance.now() - genStart;
    const generated = tokens.slice(inputIds.length);
    console.log(`Generated: [${Array.from(generated).join(', ')}]`);
    console.log(`${generated.length} tokens in ${genMs.toFixed(1)}ms (${(generated.length / genMs * 1000).toFixed(1)} tok/s)`);

    session.free();
}

main().catch(console.error);
