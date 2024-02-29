import test from 'ava';
import { Knn, TermQuery } from '../../src';

test('knn can be instantiated', t => {
    const knn = new Knn('my_field', 5, 10);
    t.truthy(knn);
});

test('knn throws error if numCandidates is less than k', t => {
    const error = t.throws(() => new Knn('my_field', 10, 5));
    t.is(error.message, 'Knn numCandidates cannot be less than k');
});

test('knn queryVector sets correctly', t => {
    const vector = [1, 2, 3];
    const knn = new Knn('my_field', 5, 10).queryVector(vector);
    t.deepEqual(knn._body.query_vector, vector);
});

test('knn queryVectorBuilder sets correctly', t => {
    const modelId = 'model_123';
    const modelText = 'Sample model text';
    const knn = new Knn('my_field', 5, 10).queryVectorBuilder(
        modelId,
        modelText
    );
    t.deepEqual(knn.query_vector_builder.text_embeddings, {
        model_id: modelId,
        model_text: modelText
    });
});

test('knn filter method adds queries correctly', t => {
    const knn = new Knn('my_field', 5, 10);
    const query = new TermQuery('field', 'value');
    knn.filter(query);
    t.deepEqual(knn._body.filter, [query]);
});

test('knn boost method sets correctly', t => {
    const boostValue = 1.5;
    const knn = new Knn('my_field', 5, 10).boost(boostValue);
    t.is(knn._body.boost, boostValue);
});

test('knn similarity method sets correctly', t => {
    const similarityValue = 0.8;
    const knn = new Knn('my_field', 5, 10).similarity(similarityValue);
    t.is(knn._body.similarity, similarityValue);
});

test('knn toJSON method returns correct DSL', t => {
    const knn = new Knn('my_field', 5, 10)
        .queryVector([1, 2, 3])
        .filter(new TermQuery('field', 'value'));

    const expectedDSL = {
        field: 'my_field',
        k: 5,
        num_candidates: 10,
        query_vector: [1, 2, 3],
        filter: [{ term: { field: 'value' } }]
    };

    t.deepEqual(knn.toJSON(), expectedDSL);
});

test('knn toJSON throws error if neither query_vector nor query_vector_builder is provided', t => {
    const knn = new Knn('my_field', 5, 10);
    const error = t.throws(() => knn.toJSON());
    t.is(
        error.message,
        'either query_vector_builder or query_vector must be provided'
    );
});
