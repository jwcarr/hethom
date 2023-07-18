if (process.argv.length != 3)
    process.exit(1);

// ------------------------------------------------------------------
// Parameters
// ------------------------------------------------------------------

// Port number to listen on
const PORT = 8080;

// Use http or https as the protocol
const PROTOCOL = 'http';

// If https, provide paths to the SSL encryption keys and Prolific credentials
const SSL_KEY_FILE = '/etc/letsencrypt/live/joncarr.net/privkey.pem';
const SSL_CERT_FILE = '/etc/letsencrypt/live/joncarr.net/fullchain.pem';
const PROLIFIC_CREDENTIALS_FILE = '/home/jon/.prolific_credentials.json';

// Regex defining valid Prolific participant IDs (24 hex digits)
const VALID_SUBJECT_ID = /^[a-f0-9]{24}$/;

// ------------------------------------------------------------------
// Import modules
// ------------------------------------------------------------------

const fs = require('fs');
const http = require(PROTOCOL);
const express = require('express');
const mongojs = require('mongojs');
const socketio = require('socket.io');

// ------------------------------------------------------------------
// Load experiment config data and Prolific credentials if available
// ------------------------------------------------------------------

const EXP_ID = process.argv[2];
const EXP_CONFIG = JSON.parse(fs.readFileSync(`config/${EXP_ID}.json`));

let PROLIFIC_CREDENTIALS = null;
if (fs.existsSync(PROLIFIC_CREDENTIALS_FILE))
	PROLIFIC_CREDENTIALS = JSON.parse(fs.readFileSync(PROLIFIC_CREDENTIALS_FILE));

// ------------------------------------------------------------------
// Server setup
// ------------------------------------------------------------------

const app = express();
app.use(express.static(`${__dirname}/client`));

const config = {};
if (PROTOCOL === 'https') {
	config.key = fs.readFileSync(SSL_KEY_FILE);
	config.cert = fs.readFileSync(SSL_CERT_FILE);
	if (PROLIFIC_CREDENTIALS) {
		app.use(express.json());
		app.post('/prolific', handleProlificEndpoint);
	}
}
const server = http.createServer(config, app);

const db = mongojs(`mongodb://127.0.0.1:27017/${EXP_ID}`, ['chains', 'subjects']);

const socket = socketio(server);

// ------------------------------------------------------------------
// Prolific endpoint
// ------------------------------------------------------------------

function submissionStatusChange(submission_id) {
	http.get(
		`https://api.prolific.co/api/v1/submissions/${submission_id}/`,
		{headers: {'Authorization': `Token ${PROLIFIC_CREDENTIALS.api_token}`}},
		(res) => {
			let raw_data = '';
			res.on('data', (chunk) => {
				raw_data += chunk;
			});
			res.on('end', () => {
				const data = JSON.parse(raw_data);
				if (data['status'] === 'RETURNED') {
					console.log(`Subject ${data.participant} has returned; dropping.`);
					db.subjects.findAndModify({
						query: {subject_id: data.participant},
						update: {$set: {status: 'dropout'}},
						new: false,
					}, (err, subject, last_err) => {
						if (err || !subject || subject.chain_id === null)
							return console.log('- Subject not found');
						db.chains.findOne({chain_id: subject.chain_id}, (err, chain) => {
							if (err || !chain)
								return console.log('- Chain not found');
							if (chain.subject_a === subject.subject_id)
								return db.chains.update({chain_id: chain.chain_id}, {'$set':{'status': 'available', 'subject_a': null}});
							else if (chain.subject_b === subject.subject_id)
								return db.chains.update({chain_id: chain.chain_id}, {'$set':{'status': 'available', 'subject_b': null}});
						});
					});
				}
			});
		}
	);
}

function handleProlificEndpoint(req, res) {
	res.sendStatus(202);
	if (req.body.event_type === 'submission.status.change')
		return submissionStatusChange(req.body.resource_id);
}

// ------------------------------------------------------------------
// Functions
// ------------------------------------------------------------------

function getCurrentTime() {
	return Math.floor(new Date() / 1000);
}

function range(start_inclusive, end_exclusive) {
	if (end_exclusive === undefined) {
		end_exclusive = start_inclusive;
		start_inclusive = 0;
	}
	const array = [];
	for (let i = start_inclusive; i < end_exclusive; i++) {
		array.push(i);
	}
	return array;
}

function randInt(start_inclusive, end_exclusive) {
	if (end_exclusive === undefined) {
		end_exclusive = start_inclusive;
		start_inclusive = 0;
	}
	return Math.floor(Math.random() * (end_exclusive - start_inclusive) + start_inclusive);
}

function shuffle(array) {
	let counter = array.length, temp, index;
	while (counter) {
		index = randInt(counter--);
		temp = array[counter];
		array[counter] = array[index];
		array[index] = temp;
	}
}

function generateItems(n, m, bottleneck=null) {
	// Return shuffled array of all index pairs for some n and m. For example,
	// if n=4 and m=4, return ['0_0', '0_1', '0_2', ..., '3_3'] in shuffled
	// order. If a bottleneck is specified, a subset of all items is returned
	// such that there is at least one of each m and at least one of each n.
	const all_items = [];
	for (let i = 0; i < n; i++) {
		for (let j = 0; j < m; j++) {
			all_items.push(`${i}_${j}`);
		}		
	}
	shuffle(all_items);
	if (bottleneck === null)
		return all_items;
	const min_dim = Math.min(m, n);
	const max_dim = Math.max(m, n);
	const indices_on_longer_dim = range(max_dim);
	shuffle(indices_on_longer_dim);
	let selected_items = [];
	for (let i=0; i < min_dim; i++) {
		const j = indices_on_longer_dim.pop();
		let item = `${i}_${j}`;
		if (n > m)
			item = `${j}_${i}`;
		all_items.splice(all_items.indexOf(item), 1);
		selected_items.push(item);
	}
	for (let j of indices_on_longer_dim) {
		const i = randInt(min_dim);
		let item = `${i}_${j}`;
		if (n > m)
			item = `${j}_${i}`;
		all_items.splice(all_items.indexOf(item), 1);
		selected_items.push(item);
	}
	const n_items_still_needed = bottleneck - selected_items.length;
	selected_items = selected_items.concat(all_items.splice(0, n_items_still_needed));
	shuffle(selected_items);
	return selected_items;
}

function generateComprehensionItems(n, m, prod_items, seen_items) {
	const comp_items = [];
	const unseen_items = [];
	for (let i = 0; i < n; i++) {
		for (let j = 0; j < m; j++) {
			const item = `${i}_${j}`;
			if (seen_items.includes(item))
				comp_items.push(item);
			else
				unseen_items.push(item);
		}		
	}
	shuffle(comp_items);
	for (let prod_item of prod_items) {
		if (unseen_items.includes(prod_item))
			comp_items.splice(randInt(prod_items.indexOf(prod_item), prod_items.length), 0, prod_item);
	}
	return comp_items;
}

function itemsWithSameWord(lexicon, target_word) {
	const compatible_items = [];
	for (let item in lexicon) {
		if (lexicon[item] === target_word)
			compatible_items.push(item);
	}
	return compatible_items;
}

function generateTrialSequenceStub(progress=null) {
	const initial_trial_estimate = 0.05;
	return [
		{event: 'consent', payload: {
			progress: 0,
		}},
		{event: 'instructions', payload: {
			instruction_screen: 'training',
			instruction_time: EXP_CONFIG.instruction_time,
			response_kind: 'next',
			progress: progress || (initial_trial_estimate * 1),
		}},
		{event: 'instructions', payload: {
			instruction_screen: 'object_prefixes',
			instruction_time: EXP_CONFIG.instruction_time,
			response_kind: 'ready_to_assign',
			progress: progress || (initial_trial_estimate * 2),
		}},
	];
}

function generateTrialSequence(task, words, training_items, lead_communicator, prod_items) {
	const trial_sequence = generateTrialSequenceStub(10);
	if (task.communication)
		trial_sequence.push({event:'instructions', payload: {
			instruction_screen: 'training_for_communication',
			instruction_time: EXP_CONFIG.instruction_time,
			response_kind: 'next',
			progress: 10,
		}});
	else
		trial_sequence.push({event:'instructions', payload: {
			instruction_screen: 'training_for_test',
			instruction_time: EXP_CONFIG.instruction_time,
			response_kind: 'next',
			progress: 10,
		}});
	const seen_items = [];
	for (let i = 0; i < task.training_reps; i++) {
		for (let j = 0; j < task.mini_test_freq; j++) {
			let training_trials = [];
			shuffle(training_items);
			for (let k = 0; k < task.bottleneck; k++) {
				const training_item = training_items[k];
				const [shape, color] = training_item.split('_');
				const training_trial = {
					item: training_item,
					word: words[training_item],
					shape: shape,
					color: color,
				};
				if (j === 0)
					seen_items.push(training_item);
				training_trials.push(training_trial);
				if (training_trials.length === task.mini_test_freq) {
					const test_item = seen_items.splice(randInt(seen_items.length),1)[0];
					const [shape, color] = test_item.split('_');
					const test_trial = {
						item: test_item,
						word: words[test_item],
						shape: shape,
						color: color,
						catch_trial: false,
						max_response_time: EXP_CONFIG.max_response_time,
					};
					trial_sequence.push({event:'training_block', payload:{
						training_trials,
						test_trial,
						exposure_time: EXP_CONFIG.exposure_time,
						pause_time: EXP_CONFIG.pause_time,
						progress: task.mini_test_freq + 1,
					}});
					training_trials = [];
				}
			}
		}
		// second to last trial on each training rep is a catch trial
		trial_sequence[trial_sequence.length - 2].payload.test_trial.catch_trial = true;
	}
	if (task.communication)
		trial_sequence.push({event:'instructions', payload: {
			instruction_screen: 'communication',
			instruction_time: EXP_CONFIG.instruction_time,
			response_kind: 'next_communication',
			progress: 10,
		}});
	else
		trial_sequence.push({event:'instructions', payload: {
			instruction_screen: 'test',
			instruction_time: EXP_CONFIG.instruction_time,
			response_kind: 'next',
			progress: 10,
		}});
	if (prod_items === undefined)
		prod_items = generateItems(task.n_shapes, task.n_colors);
	const comp_items = generateComprehensionItems(task.n_shapes, task.n_colors, prod_items, training_items);
	for (let i=0; i < prod_items.length; i++) {
		const prod_item = prod_items[i];
		const [shape, color] = prod_item.split('_');
		if (task.communication) {
			const production_event = {event:'comm_production', payload: {
				item: prod_item,
				word: words[prod_item],
				shape: shape,
				color: color,
				pause_time: EXP_CONFIG.pause_time,
				progress: 2,
			}};
			const comprehension_event = {event:'comm_comprehension', payload: {
				array: generateItems(task.n_shapes, task.n_colors),
				pause_time: EXP_CONFIG.pause_time,
				progress: 2,
			}};
			if (lead_communicator) {
				trial_sequence.push(production_event);
				trial_sequence.push(comprehension_event);
			} else {
				trial_sequence.push(comprehension_event);
				trial_sequence.push(production_event);
			}
		} else {
			trial_sequence.push({event:'test_production', payload: {
				item: prod_item,
				word: words[prod_item],
				shape: shape,
				color: color,
				pause_time: EXP_CONFIG.pause_time,
				progress: 2,
			}});
			const comp_item = comp_items[i];
			trial_sequence.push({event:'test_comprehension', payload: {
				word: words[comp_item],
				items: itemsWithSameWord(words, words[comp_item]),
				array: generateItems(task.n_shapes, task.n_colors),
				pause_time: EXP_CONFIG.pause_time,
				progress: 2,
			}});
		}

	}
	// trial_sequence.push({event:'questionnaire', payload: {
	// 	progress: 10,
	// }});
	trial_sequence.push({event:'end_of_experiment', payload: {
		return_url: EXP_CONFIG.return_url,
		basic_pay: EXP_CONFIG.basic_pay,
		progress: 0,
	}});
	let total_units_of_progress = 0;
	for (let i = 0; i < trial_sequence.length; i++) {
		total_units_of_progress += trial_sequence[i].payload.progress;
		trial_sequence[i].payload.progress = total_units_of_progress - trial_sequence[i].payload.progress;
	}
	for (let i = 0; i < trial_sequence.length; i++) {
		trial_sequence[i].payload.progress = trial_sequence[i].payload.progress / total_units_of_progress;
	}
	return [trial_sequence, comp_items];
}

function assignToChain(chains, subject_id) {
	// first look for a communicative chain where subject B has been
	// assigned but we have lost subject A for some reason - this is
	// most urgent
	for (let chain of chains) {
		if (chain.task.communication && chain.subject_a === null && chain.subject_b)
			return [chain, {$set: {status: 'unavailable', subject_a: subject_id}}];
	}
	// then look for a communicative chain where subject A has been
	// assigned but we still need a subject B
	for (let chain of chains) {
		if (chain.task.communication && chain.subject_a && chain.subject_b === null)
			return [chain, {$set: {status: 'unavailable', subject_b: subject_id}}];
	}
	// then look for a communicative chain where neither subject has been
	// assigned
	for (let chain of chains) {
		if (chain.task.communication && chain.subject_a === null && chain.subject_b === null)
			return [chain, {$set: {status: 'available', subject_a: subject_id}}];
	}
	// if no communication chains are available, return first chain in the
	// list, which will be the one with the smallest generation count
	return [chains[0], {$set: {status: 'unavailable', subject_a: subject_id}}];
}

function getPartner(client, subject, callback, n_retries=9) {
	db.chains.findOne({chain_id: subject.chain_id}, (err, chain) => {
		if (err || !chain)
			return reportError(client, 134, 'Cannot obtain chain.');
		let partner_id = null;
		if (subject.subject_id === chain.subject_a)
			partner_id = chain.subject_b;
		else if (subject.subject_id === chain.subject_b)
			partner_id = chain.subject_a;
		else
			return reportError(client, 135, 'No longer assigned to chain.');
		if (partner_id === null)
			return jiltParticipant(client, subject);
		db.subjects.findOne({subject_id: partner_id}, (err, partner) => {
			if (err || !partner)
				return reportError(client, 136, 'Assigned partner does not exist.');
			if (partner.client_id === null) {
				if (n_retries === 0)
					return reportError(client, 137, 'Unable to contact partner.');
				return setTimeout(() => {
					getPartner(client, subject, callback, n_retries - 1);
				}, 6000);
			}
			callback(partner, chain);
		});
	});
}

function prepareNextTrial(subject, sequence_position=null) {
	let next;
	if (sequence_position === null)
		next = subject.trial_sequence[subject.sequence_position];
	else
		next = subject.trial_sequence[sequence_position];
	next.payload.total_bonus = subject.total_bonus;
	next.payload.total_bonus_with_full = subject.total_bonus + EXP_CONFIG.bonus_full;
	next.payload.total_bonus_with_part = subject.total_bonus + EXP_CONFIG.bonus_part;
	if (next.event === 'end_of_experiment') {
		db.subjects.update({subject_id: subject.subject_id}, {$set: {status: 'approval_needed'}});
		db.chains.update({chain_id: subject.chain_id}, {$set: {status: 'approval_needed'}});
	}
	return next;
}

function jiltParticipant(client, subject) {
	return client.emit('report', {message: `Unfortunately, the participant you were partnered with has withdrawn from the experiment, so it will not be possible to continue. However, we will still pay you in full, including any bonus earned so far. Please click this link to return to Prolific: <a href="${EXP_CONFIG.return_url}" style="color: white">${EXP_CONFIG.return_url}</a>`});
}

function reportError(client, error_number, reason) {
	const message = 'Error ' + error_number + ': ' + reason;
	console.log(getCurrentTime() + ' ' + message);
	return client.emit('report', {message});
}

// ------------------------------------------------------------------
// Client connection handlers
// ------------------------------------------------------------------

const READY_FOR_COMMUNICATION = {}; // {subject_id: bool}
const PARTNER_COMP_ITEMS = {} // {subject_id: array}

socket.on('connection', (client) => {

	// Client makes initial handshake. If the subject has been seen before,
	// reinitialize them, otherwise, create a new subject.
	client.on('handshake', (payload) => {
		// Check for a valid subject ID
		if (!VALID_SUBJECT_ID.test(payload.subject_id))
			return reportError(client, 117, 'Invalid Prolific ID.');
		// Attempt to find the subject in the database
		db.subjects.findOne({subject_id: payload.subject_id}, (err, subject) => {
			if (err)
				return reportError(client, 118, 'Unable to validate participant ID.');
			if (subject) { // If we've seen this subject before...
				if (subject.status === 'jilted')
					return jiltParticipant(client, subject);
				if (subject.status != 'active')
					return reportError(client, 119, 'You have already completed this task.');
				// Reinitialize the subject and make a note of this in the database
				db.subjects.update({subject_id: subject.subject_id}, { $set:{client_id: client.id}, $inc: {n_reinitializations: 1} }, () => {
					return client.emit('initialize', {
						total_bonus: subject.total_bonus,
						basic_pay: EXP_CONFIG.basic_pay,
						max_pay: EXP_CONFIG.max_pay,
						session_time: EXP_CONFIG.session_time,
						object_array_dims: EXP_CONFIG.object_array_dims,
						spoken_forms: subject.spoken_forms,
					});
				});
			} else { // If we haven't seen this subject before, create a new subject
				const time = getCurrentTime();
				const subject = {
					subject_id: payload.subject_id,
					client_id: client.id,
					creation_time: time,
					modified_time: time,
					status: 'active',
					chain_id: null,
					generation: null,
					input_lexicon: null,
					spoken_forms: null,
					training_items: null,
					trial_sequence: generateTrialSequenceStub(),
					n_reinitializations: 0,
					sequence_position: 0,
					total_bonus: 0,
					responses: [],
					comments: null,
				};
				// Save the new subject to the database
				db.subjects.save(subject, (err, saved) => {
					if (err || !saved)
						return reportError(client, 120, 'Experiment unavailable. Please try again in a minute.');
					// Tell the client to initialize
					return client.emit('initialize', {
						total_bonus: subject.total_bonus,
						basic_pay: EXP_CONFIG.basic_pay,
						max_pay: EXP_CONFIG.max_pay,
						session_time: EXP_CONFIG.session_time,
						object_array_dims: EXP_CONFIG.object_array_dims,
					});
				});
			}
		});
	});

	// Client is ready to start the experiment for real. At this point, we
	// will assign the user to a chain
	client.on('ready_to_assign', (payload) => {
		// Check to see which chains are active and sort them by the number of
		// generations that have been completed (prioritizing few generations).
		db.chains.find({status: 'available'}).sort({current_gen: 1}, (err, chains) => {
			if (err || chains.length === 0)
				return reportError(client, 121, 'Experiment unavailable. Please try again in a minute.');
			// determine chain with greatest priority and update that chain
			const [candidate_chain, candidate_chain_update] = assignToChain(chains, payload.subject_id);
			db.chains.findAndModify({
				query: {chain_id: candidate_chain.chain_id, status: 'available'},
				update: candidate_chain_update,
				new: true,
			}, (err, chain, last_err) => {
				if (err || !chain)
					return reportError(client, 122, 'Unable to assign to chain');
				// if communicative chain, double check that one of the
				// subjects is the present subject ID and note whether
				// this subject is the lead subject
				let lead_communicator = null;
				let partner_id = null;
				if (chain.task.communication && chain.subject_a === payload.subject_id) {
					lead_communicator = true;
					partner_id = chain.subject_b;
				}
				else if (chain.task.communication && chain.subject_b === payload.subject_id) {
					lead_communicator = false;
					partner_id = chain.subject_a;
				}
				else if (chain.task.communication)
					return reportError(client, 123, 'Unable to assign to chain');
				// Chain assignment has been successful. Generate the
				// subject's lexicon, training items, and trial sequence, and
				// write to the database
				const input_lexicon = chain.lexicon;
				const spoken_forms = chain.spoken_forms[chain.sound_epoch];
				const training_items = generateItems(chain.task.n_shapes, chain.task.n_colors, chain.task.bottleneck);
				const [trial_sequence, comp_items] = generateTrialSequence(chain.task, input_lexicon, training_items, lead_communicator, PARTNER_COMP_ITEMS[partner_id]);
				PARTNER_COMP_ITEMS[payload.subject_id] = comp_items;
				db.subjects.findAndModify({
					query: {subject_id: payload.subject_id},
					update: {
						$set: {
							modified_time: getCurrentTime(),
							chain_id: chain.chain_id,
							generation: chain.current_gen + 1,
							input_lexicon: input_lexicon,
							spoken_forms: spoken_forms,
							training_items: training_items,
							trial_sequence: trial_sequence,
						},
						$inc: {
							sequence_position: 1,
						},
					},
					new: true,
				}, (err, subject, last_err) => {
					if (err || !subject)
						return reportError(client, 124, 'Unable to assign to chain.');
					// tell client to begin the next trial
					const next = prepareNextTrial(subject);
					next.payload.spoken_forms = spoken_forms;
					return client.emit(next.event, next.payload);
				});
			});
		});
	});

	// Client requests the next trial. Store any data and tell the client
	// which trial to run next.
	client.on('next', (payload) => {
		// Find subject in the database
		db.subjects.findOne({subject_id: payload.subject_id}, (err, subject) => {
			if (err || !subject)
				return reportError(client, 125, 'Unrecognized participant ID.');
			if (subject.status === 'jilted')
				return jiltParticipant(client, subject);
			if (subject.status != 'active')
				return reportError(client, 126, 'Your session is no longer active.');
			// Decide what information needs to be updated in the database...
			const time = getCurrentTime();
			const update = {
				$set: {modified_time: time, client_id: client.id},
				$inc: {sequence_position: 1},
			};
			if (payload.initialization)
				update.$inc.sequence_position = 0;
			if (payload.response) {
				payload.response.time = time;
				update.$push = {responses:payload.response};
				if (payload.response.test_type === 'mini_test') {
					if (payload.response.response_time < EXP_CONFIG.max_response_time) {
						if (payload.response.input_label === payload.response.expected_label)
							update.$inc.total_bonus = EXP_CONFIG.bonus_full;
						else if (payload.response.input_label.substr(0, 3) === payload.response.expected_label.substr(0, 3))
							update.$inc.total_bonus = EXP_CONFIG.bonus_part;
					}
				} else if (payload.response.test_type === 'test_production') {
					if (payload.response.input_label === payload.response.expected_label)
						update.$inc.total_bonus = EXP_CONFIG.bonus_full;
				} else if (payload.response.test_type === 'test_comprehension') {
					if (payload.response.items.includes(payload.response.selected_item))
						update.$inc.total_bonus = EXP_CONFIG.bonus_full;
				}
			}
			if (payload.comments)
				update.$set.comments = payload.comments;
			// ...and perform the update
			db.subjects.findAndModify({query: {subject_id: payload.subject_id}, update, new: true}, (err, subject, last_err) => {
				if (err || !subject)
					return reportError(client, 127, 'Unrecognized participant ID.');
				// Tell subject to begin the next trial
				let next = prepareNextTrial(subject);
				if (payload.initialization && next.event === 'comm_production' && subject.responses[subject.responses.length - 1].test_type === 'comm_production') {
					// participant appears to have interupted a communicative production trial, fast forward them to next comprehension trial
					next = prepareNextTrial(subject, subject.sequence_position + 1);
					READY_FOR_COMMUNICATION[subject.subject_id] = true;
				}
				return client.emit(next.event, next.payload);
			});
		});
	});

	// Client declares that they are ready to start a new communicative trial
	// (including the first such trial).
	client.on('next_communication', (payload) => {
		// find the subject
		const time = getCurrentTime();
		db.subjects.findAndModify({
			query: {subject_id: payload.subject_id},
			update: {
				$set: {modified_time: time, client_id: client.id},
			},
			new: false,
		}, (err, subject, last_err) => {
			if (err || !subject)
				return reportError(client, 128, 'Unrecognized participant ID.');
			if (subject.status === 'jilted')
				return jiltParticipant(client, subject);
			if (subject.status != 'active')
				return reportError(client, 129, 'Your session is no longer active.');
			// find the subject's partner
			getPartner(client, subject, (partner, chain) => {
				// if the partner has already declared themselves ready, reset
				// the partner's ready status and initiate the next trial on
				// both clients; else, mark this subject as ready. Increment
				// both subjects' sequence positions.
				if (READY_FOR_COMMUNICATION[partner.subject_id]) {
					READY_FOR_COMMUNICATION[subject.subject_id] = false;
					READY_FOR_COMMUNICATION[partner.subject_id] = false;
					const next_sequence_position = Math.min(subject.sequence_position, partner.sequence_position) + 1;
					db.subjects.update({subject_id: subject.subject_id}, {$set: {sequence_position: next_sequence_position}});
					db.subjects.update({subject_id: partner.subject_id}, {$set: {sequence_position: next_sequence_position}});
					const subject_next = prepareNextTrial(subject, next_sequence_position);
					const partner_next = prepareNextTrial(partner, next_sequence_position);
					client.emit(subject_next.event, subject_next.payload);
					client.to(partner.client_id).emit(partner_next.event, partner_next.payload);
				} else {
					READY_FOR_COMMUNICATION[subject.subject_id] = true;
				}
			});
		});
	});

	// Client sends communicative label to partner
	client.on('send_label', (payload) => {
		// Store the subject's production response
		const time = getCurrentTime();
		payload.response.time = time;
		db.subjects.findAndModify({
			query: {subject_id: payload.subject_id},
			update: {
				$set: {modified_time: time, client_id: client.id},
				$push: {responses: payload.response},
			},
			new: true,
		}, (err, subject, last_err) => {
			if (err || !subject)
				return reportError(client, 130, 'Unrecognized participant ID.');
			if (subject.status != 'active')
				return reportError(client, 131, 'Your session is no longer active.');
			// get the partner subject and forward the label to them
			getPartner(client, subject, (partner, chain) => {
				const total_bonus_with_full = partner.total_bonus + EXP_CONFIG.bonus_full;
				client.to(partner.client_id).emit('receive_message', {label: payload.response.input_label, item: payload.response.item, total_bonus_with_full, pause_time: EXP_CONFIG.pause_time});
			});
		});
	});

	// Client sends communicative feedback to partner
	client.on('send_feedback', (payload) => {
		// Store the subject's comprehension response
		const time = getCurrentTime();
		payload.response.time = time;
		db.subjects.findAndModify({
			query: {subject_id: payload.subject_id},
			update: {
				$set: {modified_time: time, client_id: client.id},
				$push: {responses: payload.response},
			},
			new: true,
		}, (err, subject, last_err) => {
			if (err || !subject)
				return reportError(client, 132, 'Unrecognized participant ID.');
			if (subject.status != 'active')
				return reportError(client, 133, 'Your session is no longer active.');
			// get the partner subject and forward the feedback to them
			getPartner(client, subject, (partner, chain) => {
				const target_item = partner.responses[partner.responses.length - 1].item;
				const selected_item = payload.response.selected_item;
				let new_partner_bonus = partner.total_bonus;
				if (selected_item === target_item) {
					db.subjects.update({subject_id: subject.subject_id}, {$inc: {total_bonus: EXP_CONFIG.bonus_full}});
					db.subjects.update({subject_id: partner.subject_id}, {$inc: {total_bonus: EXP_CONFIG.bonus_full}});
					new_partner_bonus += EXP_CONFIG.bonus_full;
				}
				client.to(partner.client_id).emit('receive_feedback', {selected_item, target_item, total_bonus: new_partner_bonus, pause_time: EXP_CONFIG.pause_time});
			});
		});
	});

	// Client has reconnected to the server, update their client ID
	client.on('reconnect', (payload) => {
		db.subjects.update({subject_id: payload.subject_id}, {$set: {client_id: client.id}});
	});

	// Client has disconnected from the server, set their client ID to null
	client.on('disconnect', () => {
		db.subjects.update({client_id: client.id}, {$set: {client_id: null}});
	});

});

server.listen(PORT);
