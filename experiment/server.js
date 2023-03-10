// ------------------------------------------------------------------
// Parameters
// ------------------------------------------------------------------

// Name used for the MongoDB database
const EXP_ID = 'pilot1';

// Regex defining valid Prolific participant IDs (24 hex digits)
const VALID_SUBJECT_ID = /^[a-f0-9]{24}$/;

// Port number to listen on
const PORT = 8080;

// Use http or https as the protocol
const PROTOCOL = 'http';

// If https, provide paths to the SSL encryption keys
const SSL_KEY_FILE = '/etc/letsencrypt/live/joncarr.net/privkey.pem';
const SSL_CERT_FILE = '/etc/letsencrypt/live/joncarr.net/fullchain.pem';

// ------------------------------------------------------------------
// Import modules
// ------------------------------------------------------------------

const fs = require('fs');
const http = require(PROTOCOL);
const express = require('express');
const mongojs = require('mongojs');
const socketio = require('socket.io');

// ------------------------------------------------------------------
// Load experiment config data
// ------------------------------------------------------------------

const EXP_CONFIG = JSON.parse(fs.readFileSync(`config/${EXP_ID}.json`));

// ------------------------------------------------------------------
// Server setup
// ------------------------------------------------------------------

const app = express();
app.use(express.static(`${__dirname}/client`));

const config = {};
if (PROTOCOL === 'https') {
	config.key = fs.readFileSync(SSL_KEY_FILE);
	config.cert = fs.readFileSync(SSL_CERT_FILE);
}
const server = http.createServer(config, app);

const db = mongojs(`mongodb://127.0.0.1:27017/${EXP_ID}`, ['chains', 'subjects']);

const socket = socketio(server);

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
		if (n < m)
			item = `${j}_${i}`;
		all_items.splice(all_items.indexOf(item), 1);
		selected_items.push(item);
	}
	for (let j of indices_on_longer_dim) {
		const i = randInt(min_dim);
		let item = `${i}_${j}`;
		if (n < m)
			item = `${j}_${i}`;
		all_items.splice(all_items.indexOf(item), 1);
		selected_items.push(item);
	}
	const n_items_still_needed = bottleneck - selected_items.length;
	selected_items = selected_items.concat(all_items.splice(0, n_items_still_needed));
	shuffle(selected_items);
	return selected_items;
}

function generateTrialSequence(task, words, trained_item_indices) {
	const seen_items = [], trial_sequence = [];
	trial_sequence.push({event: 'consent', payload:{progress: 0}});
	trial_sequence.push({event: 'training_instructions', payload:{progress: 0}});
	for (let i = 0; i < task.training_reps; i++) {
		for (let j = 0; j < task.mini_test_freq; j++) {
			let training_trials = [];
			shuffle(trained_item_indices);
			for (let k = 0; k < task.bottleneck; k++) {
				const training_item = trained_item_indices[k];
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
					};
					trial_sequence.push({event:'training_block', payload:{
						training_trials,
						test_trial,
						trial_time: task.trial_time,
						pause_time: task.pause_time,
						progress: task.mini_test_freq + 1,
						bonus_partial: task.bonus_partial,
						bonus_full: task.bonus_full,
					}});
					training_trials = [];
				}
			}
		}
		// second to last trial on each training rep is a catch trial
		trial_sequence[trial_sequence.length - 2].payload.test_trial.catch_trial = true;
	}
	trial_sequence.push({event:'test_instructions', payload:{
		instruction_time : task.instruction_time,
		progress : 10,
	}});
	const prod_item_indices = generateItems(task.n_shapes, task.n_colors);
	const comp_item_indices = generateItems(task.n_shapes, task.n_colors);
	for (let i=0; i < prod_item_indices.length; i++) {
		const prod_item = prod_item_indices[i];
		const [shape, color] = prod_item.split('_');
		trial_sequence.push({event:'test_production', payload:{
			item: prod_item,
			word: words[prod_item],
			shape: shape,
			color: color,
			pause_time: task.pause_time,
			progress: 2,
			bonus_partial: task.bonus_partial,
			bonus_full: task.bonus_full,
		}});
		const comp_item = comp_item_indices[i];
		trial_sequence.push({event:'test_comprehension', payload:{
			item: comp_item,
			word: words[comp_item],
			array: generateItems(task.n_shapes, task.n_colors),
			pause_time: task.pause_time,
			progress: 2,
			bonus_partial: task.bonus_partial,
			bonus_full: task.bonus_full,
		}});
	}
	trial_sequence.push({event:'questionnaire', payload:{
		progress : 10,
	}});
	trial_sequence.push({event:'end_of_experiment', payload:{
		return_url : task.return_url,
		basic_pay : task.basic_pay,
		progress : 0,
	}});
	let total_units_of_progress = 0;
	for (let i = 0; i < trial_sequence.length; i++) {
		total_units_of_progress += trial_sequence[i].payload.progress;
		trial_sequence[i].payload.progress = total_units_of_progress - trial_sequence[i].payload.progress;
	}
	for (let i = 0; i < trial_sequence.length; i++) {
		trial_sequence[i].payload.progress = trial_sequence[i].payload.progress / total_units_of_progress;
	}
	return trial_sequence;
}

function reportError(client, error_number, reason) {
	const message = 'Error ' + error_number + ': ' + reason;
	console.log(getCurrentTime() + ' ' + message);
	return client.emit('report', {message});
}

// ------------------------------------------------------------------
// Client connection handlers
// ------------------------------------------------------------------

socket.on('connection', function(client) {

	// Client makes initial handshake. Check if we've seen them before, if so
	client.on('handshake', function(payload) {
		// Check for a valid subject ID
		if (!VALID_SUBJECT_ID.test(payload.subject_id))
			return reportError(client, 117, 'Unable to validate participant ID.');
		// Attempt to find the subject in the database
		db.subjects.findOne({subject_id: payload.subject_id}, function(err, subject) {
			if (err)
				return reportError(client, 118, 'Unable to validate participant ID.');
			if (subject) { // If we've seen this subject before...
				if (subject.status != 'active')
					return reportError(client, 116, 'You have already completed this task.');
				// Reinitialize the subject and make a note of this in the database
				db.subjects.update({subject_id: subject.subject_id}, { $set:{client_id: client.id}, $inc: {n_reinitializations: 1} }, function() {
					return client.emit('initialize', {
						total_bonus: subject.total_bonus,
						basic_pay: subject.basic_pay,
						max_pay: EXP_CONFIG.max_pay,
						session_time: EXP_CONFIG.session_time,
						object_array_dims: EXP_CONFIG.object_array_dims,
					});
				});
			} else { // If we haven't seen this subject before, create a new subject
				const time = getCurrentTime();
				const subject = {
					subject_id: payload.subject_id,
					client: client.id,
					creation_time: time,
					modified_time: time,
					status: 'active',
					chain_id: null,
					generation: null,
					input_lexicon: null,
					training_items: null,
					trial_sequence: [
						{event: 'consent', payload:{}},
						{event: 'training_instructions', payload:{}},
					],
					basic_pay: EXP_CONFIG.basic_pay,
					bonus_partial: EXP_CONFIG.bonus_partial,
					bonus_full: EXP_CONFIG.bonus_full,
					n_reinitializations: 0,
					sequence_position: 0,
					total_bonus: 0,
					responses: [],
					comments: null,
				};
				// Save the new subject to the database
				db.subjects.save(subject, function(err, saved) {
					if (err || !saved)
						return reportError(client, 121, 'This task is currently unavailable.');
					// Tell the client to initialize
					return client.emit('initialize', {
						total_bonus: subject.total_bonus,
						basic_pay: subject.basic_pay,
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
	client.on('ready', function(payload) {
		// Check to see which chains are active and sort them by the number of
		// participants that have taken part so far.
		db.chains.find({status: 'available'}).sort({current_gen: 1}, function(err, chains) {
			if (err || chains.length === 0)
				return reportError(client, 119, 'Experiment unavailable. Please try again in a minute.');
			const chain = chains[0];
			db.chains.update({chain_id: chain.chain_id}, {$set: {status: 'unavailable'}}, function() {
				const input_lexicon = chain.lexicon;
				const training_items = generateItems(chain.task.n_shapes, chain.task.n_colors, chain.task.bottleneck);
				const trial_sequence = generateTrialSequence(chain.task, input_lexicon, training_items);
				db.subjects.findAndModify({
					query: {subject_id: payload.subject_id},
					update: {
						$set: {
							modified_time: getCurrentTime(),
							chain_id: chain.chain_id,
							generation: chain.current_gen + 1,
							input_lexicon: input_lexicon,
							training_items: training_items,
							trial_sequence: trial_sequence,
						},
						$inc: {
							sequence_position: 1,
						},
					},
					new: true,
				}, function(err, subject, last_err) {
					if (err || !subject)
						return reportError(client, 128, 'Unrecognized participant ID.');
					const next = subject.trial_sequence[subject.sequence_position];
					next.payload.total_bonus = subject.total_bonus;
					return client.emit(next.event, next.payload);
				});
			});
		});
	});

	// Client requests the next trial. Store any data and tell the client
	// which trial to run next.
	client.on('next', function(payload) {
		const time = getCurrentTime();
		db.subjects.findOne({subject_id: payload.subject_id}, function(err, subject) {
			if (err || !subject)
				return reportError(client, 126, 'Unrecognized participant ID.');
			if (subject.status != 'active')
				return reportError(client, 127, 'Your session is no longer active.');
			const update = {
				$set : {modified_time: time, client_id: client.id},
				$inc : {sequence_position: 1},
			};
			if (payload.initialization)
				update.$inc.sequence_position = 0;
			if (payload.response) {
				payload.response.time = time;
				update.$push = {responses:payload.response};
				if (payload.response.test_type === 'mini_test') {
					if (payload.response.input_label === payload.response.expected_label)
						update.$inc.total_bonus = subject.bonus_full;
					else if (payload.response.input_label.substr(0, 3) === payload.response.expected_label.substr(0, 3))
						update.$inc.total_bonus = subject.bonus_partial;
				} else if (payload.response.test_type === 'test_production') {
					if (payload.response.input_label === payload.response.expected_label)
						update.$inc.total_bonus = subject.bonus_full;
				} else if (payload.response.test_type === 'test_comprehension') {
					if (payload.response.selected_item === payload.response.item)
						update.$inc.total_bonus = subject.bonus_full;
				}
			}
			if (payload.comments)
				update.$set.comments = payload.comments;
			db.subjects.findAndModify({query: {subject_id: payload.subject_id}, update, new: true}, function(err, subject, last_err) {
				if (err || !subject)
					return reportError(client, 128, 'Unrecognized participant ID.');
				const next = subject.trial_sequence[subject.sequence_position];
				next.payload.total_bonus = subject.total_bonus;
				if (next.event === 'end_of_experiment') {
					db.subjects.update({subject_id: subject.subject_id}, {$set: {status: 'approval_needed'}});
					db.chains.update({chain_id: subject.chain_id}, {$set: {status: `approval_needed ${subject.subject_id}`}});
				}
				return client.emit(next.event, next.payload);
			});
		});
	});

	client.on('disconnect', function() {
		db.subjects.update({client_id: client.id}, {$set: {client_id: null}});
	});

});

server.listen(PORT);
