import type { BaseLanguageModel } from '@langchain/core/language_models/base';
import { HumanMessage } from '@langchain/core/messages';
import {
	SystemMessagePromptTemplate,
	ChatPromptTemplate,
	HumanMessagePromptTemplate,
} from '@langchain/core/prompts';
import { OutputFixingParser, StructuredOutputParser } from 'langchain/output_parsers';
import { NodeOperationError, NodeConnectionTypes } from 'n8n-workflow';
import type {
	IDataObject,
	IExecuteFunctions,
	INodeExecutionData,
	INodeParameters,
	INodeType,
	INodeTypeDescription,
} from 'n8n-workflow';
import { z } from 'zod';

import { getTracingConfig } from '@utils/tracing';

const SYSTEM_PROMPT_TEMPLATE =
	"Please classify the text provided by the user into one of the following categories: {categories}, and use the provided formatting instructions below. Don't explain, and only output the json.";

const configuredOutputs = (parameters: INodeParameters) => {
	if (parameters.loadCategoriesFromInputItems) {
		return [{ type: 'main', displayName: 'Output' }];
	}
	const categories = ((parameters.categories as IDataObject)?.categories as IDataObject[]) ?? [];
	const fallback = (parameters.options as IDataObject)?.fallback as string;
	const ret = categories.map((cat) => {
		return { type: 'main', displayName: cat.category };
	});
	if (fallback === 'other') ret.push({ type: 'main', displayName: 'Other' });
	return ret;
};

export class TextClassifier implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Text Classifier',
		name: 'textClassifier',
		icon: 'fa:tags',
		iconColor: 'black',
		group: ['transform'],
		version: 1,
		description: 'Classify your text into distinct categories',
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Chains', 'Root Nodes'],
			},
			resources: {
				primaryDocumentation: [
					{
						url: 'https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.text-classifier/',
					},
				],
			},
		},
		defaults: {
			name: 'Text Classifier',
		},
		inputs: [
			{ displayName: '', type: NodeConnectionTypes.Main },
			{
				displayName: 'Model',
				maxConnections: 1,
				type: NodeConnectionTypes.AiLanguageModel,
				required: true,
			},
		],
		outputs: `={{(${configuredOutputs})($parameter)}}`,
		properties: [
			{
				displayName: 'Load Categories from Input Items',
				name: 'loadCategoriesFromInputItems',
				type: 'boolean',
				default: false,
				description:
					'If enabled, categories will be loaded from the input items (each item must have a "category" and "description" field)',
			},
			{
				displayName: 'Categories',
				name: 'categories',
				placeholder: 'Add Category',
				type: 'fixedCollection',
				default: {},
				typeOptions: {
					multipleValues: true,
				},
				options: [
					{
						name: 'categories',
						displayName: 'Categories',
						values: [
							{
								displayName: 'Category',
								name: 'category',
								type: 'string',
								default: '',
								description: 'Category to add',
								required: true,
							},
							{
								displayName: 'Description',
								name: 'description',
								type: 'string',
								default: '',
								description: "Describe your category if it's not obvious",
							},
						],
					},
				],
				displayOptions: {
					show: {
						loadCategoriesFromInputItems: [false],
					},
				},
			},
			{
				displayName: 'Text to Classify',
				name: 'inputText',
				type: 'string',
				required: true,
				default: '',
				description: 'Use an expression to reference data in previous nodes or enter static text',
				typeOptions: {
					rows: 2,
				},
			},
			{
				displayName: 'Options',
				name: 'options',
				type: 'collection',
				default: {},
				placeholder: 'Add Option',
				options: [
					{
						displayName: 'Allow Multiple Classes To Be True',
						name: 'multiClass',
						type: 'boolean',
						default: false,
					},
					{
						displayName: 'When No Clear Match',
						name: 'fallback',
						type: 'options',
						default: 'discard',
						description: "What to do with items that don't match the categories exactly",
						options: [
							{
								name: 'Discard Item',
								value: 'discard',
								description: 'Ignore the item and drop it from the output',
							},
							{
								name: "Output on Extra, 'Other' Branch",
								value: 'other',
								description: "Create a separate output branch called 'Other'",
							},
						],
					},
					{
						displayName: 'System Prompt Template',
						name: 'systemPromptTemplate',
						type: 'string',
						default: SYSTEM_PROMPT_TEMPLATE,
						description: 'String to use directly as the system prompt template',
						typeOptions: {
							rows: 6,
						},
					},
					{
						displayName: 'Enable Auto-Fixing',
						name: 'enableAutoFixing',
						type: 'boolean',
						default: true,
						description:
							'Whether to enable auto-fixing (may trigger an additional LLM call if output is broken)',
					},
				],
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();

		const llm = (await this.getInputConnectionData(
			NodeConnectionTypes.AiLanguageModel,
			0,
		)) as BaseLanguageModel;

		const loadCategoriesFromInputItems = this.getNodeParameter(
			'loadCategoriesFromInputItems',
			0,
			false,
		) as boolean;

		// Get categories (branch-specific)
		let categories: Array<{ category: string; description: string }>;

		if (loadCategoriesFromInputItems) {
			// Aggregate all categories from input items
			categories = items.map((item, idx) => {
				const category = item.json.category;
				const description = item.json.description;
				if (typeof category !== 'string' || typeof description !== 'string') {
					throw new NodeOperationError(
						this.getNode(),
						`Input item ${idx} is missing a valid 'category' or 'description' field.`,
					);
				}
				return { category, description };
			});

			if (!categories || categories.length === 0) {
				throw new NodeOperationError(this.getNode(), 'At least one category must be defined');
			}

			const input = this.getNodeParameter('inputText', 0) as string;
			if (input === undefined || input === null) {
				throw new NodeOperationError(this.getNode(), 'Text to classify is not defined');
			}

			// Common options fetching
			const options = this.getNodeParameter('options', 0, {}) as {
				multiClass: boolean;
				fallback?: string;
				systemPromptTemplate?: string;
				enableAutoFixing: boolean;
			};
			const multiClass = options?.multiClass ?? false;
			const fallback = options?.fallback ?? 'discard';
			const systemPromptTemplateOpt = this.getNodeParameter(
				'options.systemPromptTemplate',
				0,
				SYSTEM_PROMPT_TEMPLATE,
			) as string;

			// Common schema creation
			const schemaEntries = categories.map((cat) => [
				cat.category,
				z
					.boolean()
					.describe(
						`Should be true if the input has category "${cat.category}" (description: ${cat.description})`,
					),
			]);
			if (fallback === 'other')
				schemaEntries.push([
					'fallback',
					z.boolean().describe('Should be true if none of the other categories apply'),
				]);
			const schema = z.object(Object.fromEntries(schemaEntries));

			const structuredParser = StructuredOutputParser.fromZodSchema(schema);
			const parser = options.enableAutoFixing
				? OutputFixingParser.fromLLM(llm, structuredParser)
				: structuredParser;

			const multiClassPrompt = multiClass
				? 'Categories are not mutually exclusive, and multiple can be true'
				: 'Categories are mutually exclusive, and only one can be true';

			const fallbackPrompt = {
				other: 'If no categories apply, select the "fallback" option.',
				discard: 'If there is not a very fitting category, select none of the categories.',
			}[fallback];

			// Common prompt template and chain
			const prompt = ChatPromptTemplate.fromMessages([
				SystemMessagePromptTemplate.fromTemplate(
					`${systemPromptTemplateOpt ?? SYSTEM_PROMPT_TEMPLATE}
{format_instructions}
${multiClassPrompt}
${fallbackPrompt}`,
				),
				HumanMessagePromptTemplate.fromTemplate('{inputText}'),
			]);

			const chain = prompt.pipe(llm).pipe(parser).withConfig(getTracingConfig(this));

			// Branch-specific execution and output
			try {
				const output = await chain.invoke({
					categories: categories.map((cat) => cat.category).join(', '),
					format_instructions: parser.getFormatInstructions(),
					inputText: input,
				});
				// Output a single item with the classification result
				return [[{ json: { ...output, input } }]];
			} catch (error) {
				return [[{ json: { error: (error as Error).message } }]];
			}
		} else {
			categories = this.getNodeParameter('categories.categories', 0, []) as Array<{
				category: string;
				description: string;
			}>;

			if (!categories || categories.length === 0) {
				throw new NodeOperationError(this.getNode(), 'At least one category must be defined');
			}

			// Common options fetching
			const options = this.getNodeParameter('options', 0, {}) as {
				multiClass: boolean;
				fallback?: string;
				systemPromptTemplate?: string;
				enableAutoFixing: boolean;
			};
			const multiClass = options?.multiClass ?? false;
			const fallback = options?.fallback ?? 'discard';
			const systemPromptTemplateOpt = this.getNodeParameter(
				'options.systemPromptTemplate',
				0,
				SYSTEM_PROMPT_TEMPLATE,
			) as string;

			// Common schema creation
			const schemaEntries = categories.map((cat) => [
				cat.category,
				z
					.boolean()
					.describe(
						`Should be true if the input has category "${cat.category}" (description: ${cat.description})`,
					),
			]);
			if (fallback === 'other')
				schemaEntries.push([
					'fallback',
					z.boolean().describe('Should be true if none of the other categories apply'),
				]);
			const schema = z.object(Object.fromEntries(schemaEntries));

			const structuredParser = StructuredOutputParser.fromZodSchema(schema);
			const parser = options.enableAutoFixing
				? OutputFixingParser.fromLLM(llm, structuredParser)
				: structuredParser;

			const multiClassPrompt = multiClass
				? 'Categories are not mutually exclusive, and multiple can be true'
				: 'Categories are mutually exclusive, and only one can be true';

			const fallbackPrompt = {
				other: 'If no categories apply, select the "fallback" option.',
				discard: 'If there is not a very fitting category, select none of the categories.',
			}[fallback];

			// Common prompt template and chain
			const prompt = ChatPromptTemplate.fromMessages([
				SystemMessagePromptTemplate.fromTemplate(
					`${systemPromptTemplateOpt ?? SYSTEM_PROMPT_TEMPLATE}
{format_instructions}
${multiClassPrompt}
${fallbackPrompt}`,
				),
				HumanMessagePromptTemplate.fromTemplate('{inputText}'),
			]);

			const chain = prompt.pipe(llm).pipe(parser).withConfig(getTracingConfig(this));

			const returnData: INodeExecutionData[][] = Array.from(
				{ length: categories.length + (fallback === 'other' ? 1 : 0) },
				(_) => [],
			);

			// Process each input item
			for (let itemIdx = 0; itemIdx < items.length; itemIdx++) {
				const item = items[itemIdx];
				item.pairedItem = { item: itemIdx };
				const input = this.getNodeParameter('inputText', itemIdx) as string;

				if (input === undefined || input === null) {
					if (this.continueOnFail()) {
						returnData[0].push({
							json: { error: 'Text to classify is not defined' },
							pairedItem: { item: itemIdx },
						});
						continue;
					} else {
						throw new NodeOperationError(
							this.getNode(),
							`Text to classify for item ${itemIdx} is not defined`,
						);
					}
				}

				try {
					// Only call the LLM ONCE per input item
					const output = await chain.invoke({
						categories: categories.map((cat) => cat.category).join(', '),
						format_instructions: parser.getFormatInstructions(),
						inputText: input,
					});

					categories.forEach((cat, idx) => {
						if (output[cat.category]) returnData[idx].push(item);
					});
					if (fallback === 'other' && output.fallback) returnData[returnData.length - 1].push(item);
				} catch (error) {
					if (this.continueOnFail()) {
						returnData[0].push({
							json: { error: error.message },
							pairedItem: { item: itemIdx },
						});
						continue;
					}

					throw error;
				}
			}
			return returnData;
		}
	}
}
