// Establish websocket connection with the server using socket.io
const socket = io.connect();

// Extract Prolific ID from the URL
const subject_id = (new URL(window.location.href)).searchParams.get('PROLIFIC_PID');

const word_audio = [
	new Audio('sounds/0.mp4'),
	new Audio('sounds/1.mp4'),
	new Audio('sounds/2.mp4'),
	new Audio('sounds/3.mp4'),
];

const test_audio = [
	[new Audio('sounds/test0.mp4'), 'the quick cat'],
	[new Audio('sounds/test1.mp4'), 'the green bird'],
	[new Audio('sounds/test2.mp4'), 'the hungry dog'],
	[new Audio('sounds/test3.mp4'), 'the angry lion'],
];

const catch_instruction = new Audio('sounds/catch_instruction.mp4');

function iterAtInterval(iterable, interval, func, final_func) {
	// Call func on each item in an iterable with a given time interval. Once all
	// items have been iterated over, call final_func.
	if (iterable.length === 0) {
		final_func();
	} else {
		setTimeout(function() {
			iterAtInterval(iterable.slice(1), interval, func, final_func);
		}, interval);
		func(iterable[0]);
	}
}

function randInt(start_inclusive, end_exclusive) {
	if (end_exclusive === undefined) {
		end_exclusive = start_inclusive;
		start_inclusive = 0;
	}
	return Math.floor(Math.random() * (end_exclusive - start_inclusive) + start_inclusive);
}

function updateProgress(progress) {
	$('#progress').animate({width: progress*900}, 250);
}

function updateBonus(total_bonus) {
	$('#bonus_amount').html(total_bonus);
}

function preloadObject(shape, color) {
	$('#object_image').attr('src', `images/shapes/${shape}_${color}.png`);
}

function showObject() {
	$('#object').show();
}

function hideObject() {
	$('#object').hide();
	$('#object_image').off('click');
	$('#object_image').css('cursor', 'default');
}

function preloadArray(object_array) {
	for (let i=0; i < object_array.length; i++) {
		$(`#object_array_${i}`).attr('src', `images/shapes/${object_array[i]}.png`);
		$(`#object_array_${i}`).css('cursor', 'pointer');
	}
}

function showArray() {
	$('#object_array').show();
}

function hideArray() {
	$('#object_array').hide();
	$('img[id^="object_array_"]').css('opacity', '1');
}

function showWord(word) {
	$('#word').html(word);
	$('#word').show();
}

function playWord(sound) {
	word_audio[sound].play();
}

function hideWord() {
	$('#word').hide();
}

function showLabelInput() {
	$('#label').val('');
	$('#label_input').show();
	$('#label').focus();
}

function hideLabelInput() {
	$('#label_input').hide();
}

function enableButton(button_id) {
	$(button_id).css('background-color', 'black');
	$(button_id).css('cursor', 'pointer');
	$(button_id).attr('disabled', false);
}

function disableButton(button_id) {
	$(button_id).css('background-color', 'gray');
	$(button_id).css('cursor', 'default');
	$(button_id).attr('disabled', true);
}

function showInputError(input_id) {
	$(input_id).css("background-color", "#F5E3E6");
	setTimeout(function() {
		$(input_id).css("background-color", "#FFFFFF");
	}, 500);
}

function validateWord(label, expected_label=null) {
	if (!label.match(/^[a-z]{4,9}$/))
		return false;
	if (expected_label) {
		if (label.slice(0, 3) != expected_label.slice(0, 3))
			return false;
	}
	return true;
}

function diffLabels(s1, s2) {
	const sequence_matchcer = new difflib.SequenceMatcher(null, s1, s2);
	const red = '<span style="color: red; text-decoration: line-through;">';
	const green = '<span style="color: green; font-weight:bold;">';
	const close = '</span>';
	let feedback = '';
	for (let [tag, i1, i2, j1, j2] of sequence_matchcer.getOpcodes()) {
		if (tag === 'equal') {
			feedback += s1.slice(i1, i2);
		}
		else if (tag === 'delete') {
			feedback += red + s1.slice(i1, i2) + close;
		}
		else if (tag === 'insert') {
			feedback += green + s2.slice(j1, j2) + close;
		}
		else if (tag === 'replace') {
			feedback += red + s1.slice(i1, i2) + close;
			feedback += green + s2.slice(j1, j2) + close;
		}
	}
	return feedback;
}

function initializeObjectArray(object_array_dims) {
	const [m, n] = object_array_dims;
	const size = Math.floor(Math.min(350 / m, 900 / n));
	let html = '<table>';
	let position = 0;
	for (let i=0; i < m; i++) {
		html += '<tr>';
		for (let j=0; j < n; j++) {
			html += '<td>';
			html += `<img id='object_array_${position}' src='images/empty_char.png' width='${size}' height='${size}' />`;
			html += '</td>';
			position++;
		}
		html += '</tr>';
	}
	html += '</table>';
	$('#object_array').html(html);
}

socket.on('initialize', function(payload) {
	updateBonus(payload.total_bonus);
	initializeObjectArray(payload.object_array_dims);
	socket.emit('next', {subject_id, 'initialization': true});
});

socket.on('consent', function(payload) {
	$('#consent_session_time').html(payload.session_time);
	$('#consent_basic_pay').html('£' + (payload.basic_pay/100).toFixed(2));
	$('#consent_max_pay').html('£' + (payload.max_pay/100).toFixed(2));
	$('#submit_consent').click(function() {
		$('#submit_consent').off('click');
		$('#consent_screen').hide();
		socket.emit('next', {subject_id});
	});
	$('#confirm_consent').click(function() {
		if ($('#confirm_consent').is(':checked'))
			enableButton('#submit_consent');
		else
			disableButton('#submit_consent');
	});
	disableButton('#submit_consent');
	$('#consent_screen').show();
});

socket.on('training_instructions', function(payload) {
	updateProgress(payload.progress);
	$('#start_training').click(function() {
		$('#start_training').off('click');
		$('#training_instructions').hide();
		socket.emit('next', {subject_id});
	});
	setTimeout(function() {
		enableButton('#start_training');
	}, payload.instruction_time);
	disableButton('#start_training');
	$('#training_instructions').show();
});

socket.on('test_instructions', function(payload) {
	$('#experiment').hide();
	updateProgress(payload.progress);
	$('#start_test').click(function() {
		$('#start_test').off('click');
		$('#test_instructions').hide();
		$('#experiment').show();
		socket.emit('next', {subject_id});
	});
	setTimeout(function() {
		enableButton('#start_test');
	}, payload.instruction_time);
	disableButton('#start_test');
	$('#test_instructions').show();
});

socket.on('training_block', function(payload) {
	console.log(payload);
	updateProgress(payload.progress);
	iterAtInterval(payload.training_trials, payload.trial_time,
		// func: On each passive exposure trial...
		function(trial) {
			// 1. Preload the object and word
			hideObject();
			hideWord();
			preloadObject(trial.shape, trial.color);
			setTimeout(function() {
				setTimeout(function() {
					// 3. After pause_time, show the word
					playWord(trial.shape);
					showWord(trial.word);
				}, payload.pause_time);
				// 2. After pause_time, show the object
				showObject();
			}, payload.pause_time);
		},
		// final_func: On each mini-test trial...
		function() {
			// 1. Preload the test word
			hideObject();
			hideWord();
			preloadObject(payload.test_trial.shape, payload.test_trial.color);
			let object_clicked = false;
			$('#object_image').click(function() {
				playWord(payload.test_trial.shape);
				$('#label').focus();
				object_clicked = true;
			}).css('cursor', 'pointer');
			setTimeout(function() {
				$("#input_form").submit(function(event) {
					event.preventDefault();
					let label = $("#label").val();
					if (validateWord(label)) {
						$("#input_form").off('submit');
						const response_time = Math.floor(performance.now() - start_time);
						setTimeout(function() {
							// 4. After 2*pause_time, hide the word and object buttons and request the next trial
							hideObject();
							socket.emit('next', {subject_id, response: {
								test_type: 'mini_test',
								shape: payload.test_trial.shape,
								color: payload.test_trial.color,
								expected_label: payload.test_trial.word,
								catch_trial: payload.test_trial.catch_trial,
								input_label: label,
								response_time,
								object_clicked,
							}});
						}, payload.pause_time * 2);
						// 3. On enter, show feedback and update the user's bonus
						hideLabelInput();
						showWord(diffLabels(label, payload.test_trial.word));
						if (label === payload.test_trial.word)
							updateBonus(payload.total_bonus + payload.bonus_full);
						else if (label.substr(0, 3) === payload.test_trial.word.substr(0, 3))
							updateBonus(payload.total_bonus + payload.bonus_partial);

					} else {
						showInputError('#label');
					}
					return false;
				});
				// 2. After pause_time, show the object and input box
				showObject();
				showLabelInput();
				if (payload.test_trial.catch_trial)
					catch_instruction.play();
				const start_time = performance.now();
			}, payload.pause_time);
		}
	);
	$('#experiment').show();
});

socket.on('test_production', function(payload) {
	console.log(payload);
	updateProgress(payload.progress);
	// 1. Preload the test word
	hideObject();
	hideWord();
	preloadObject(payload.shape, payload.color);
	$('#object_image').click(function() {
		playWord(payload.shape);
		$('#label').focus();
	}).css('cursor', 'pointer');
	setTimeout(function() {
		$("#input_form").submit(function(event) {
			event.preventDefault();
			let label = $("#label").val();
			if (validateWord(label, payload.word)) {
				// 3. On enter, update the user's bonus
				$("#input_form").off('submit');
				const response_time = Math.floor(performance.now() - start_time);
				hideLabelInput();
				hideObject();
				if (label === payload.word)
					updateBonus(payload.total_bonus + payload.bonus_full);
				socket.emit('next', {subject_id, response: {
					test_type: 'test_production',
					shape: payload.shape,
					color: payload.color,
					expected_label: payload.word,
					input_label: label,
					response_time,
				}});
				// setTimeout(function() {
					// 4. After 2*pause_time, hide the word and object buttons and request the next trial
				// }, payload.pause_time * 2);
			} else {
				playWord(payload.shape);
				showInputError('#label');
			}
			return false;
		});
		// 2. After pause_time, show the object and input box
		showObject();
		showLabelInput();
		const start_time = performance.now();
	}, payload.pause_time);
	$('#experiment').show();
});

socket.on('test_comprehension', function(payload) {
	console.log(payload);
	updateProgress(payload.progress);
	// 1. Preload the test word
	hideArray();
	hideWord();
	preloadArray(payload.array);
	setTimeout(function() {
		$('img[id^="object_array_"]').click(function() {
			$('img[id^="object_array_"]').off('click');
			$('img[id^="object_array_"]').css('cursor', 'default');
			const response_time = Math.floor(performance.now() - start_time);
			const selected_button = parseInt($(this).attr('id').match(/object_array_(.+)/)[1]);
			const selected_item = payload.array[selected_button];
			// 3. object clicked, hide array and move on
			hideArray();
			socket.emit('next', {subject_id, response: {
				test_type: 'test_comprehension',
				item: payload.item,
				word: payload.word,
				selected_button,
				selected_item,
				response_time,
			}});
			///////////////////////////////////
			// FOR COMMUNICATION ONLY - feedback
			// setTimeout(function() {
			//   3. On enter, show feedback and update the user's bonus
			// }, payload.pause_time * 2);
			// const correct_object_position = payload.array.indexOf(payload.item);
			// for (let i in payload.array) {
			// 	if (i != correct_object_position)
			// 		$(`#object_array_${i}`).css('opacity', '0.1');
			// }
			///////////////////////////////////
			if (selected_item === payload.item)
				updateBonus(payload.total_bonus + payload.bonus_full);
		});
		// 2. After pause_time, show the object and input box
		showWord(payload.word);
		showArray();
		const start_time = performance.now();
	}, payload.pause_time);
	$('#experiment').show();
});

socket.on('questionnaire', function(payload) {
	updateProgress(payload.progress);
	$('#experiment').hide();
	$('#submit_questionnaire').click(function() {
		$('#submit_questionnaire').off('click');
		const comments = $('#comments').val();
		socket.emit('next', {subject_id, comments});
	});
	$('#comments').keyup(function() {
		if ($(this).val().length > 0)
			enableButton('#submit_questionnaire');
		else
			disableButton('#submit_questionnaire');
	});
	disableButton('#submit_questionnaire');
	$('#questionnaire').show();
	$('#comments').focus();
});

socket.on('end_of_experiment', function(payload) {
	updateProgress(payload.progress);
	$('#questionnaire').hide();
	$('#basic_pay').html('£' + (payload.basic_pay/100).toFixed(2));
	$('#bonus_pay').html('£' + (payload.total_bonus/100).toFixed(2));
	$('#total_pay').html('£' + ((payload.basic_pay + payload.total_bonus)/100).toFixed(2));
	$('#exit').click(function() {
		$('#exit').off('click');
		$('#exit').hide();
		window.location.href = payload.return_url;
	});
	$('#end_of_experiment').show();
});

socket.on('report', function(payload) {
	$('#status_message').html(payload.message);
	$('#task_status').show();
});

$(document).ready(function() {
	if (/Android|webOS|iPhone|iPad|iPod|IEMobile|Opera Mini/i.test(navigator.userAgent))
		return null;
	const [test_sound, test_phrase] = test_audio[randInt(test_audio.length)];
	$('#test_sound_input').keyup(function() {
		if ($(this).val().toLowerCase() === test_phrase) {
			$('#test_sound_input').off('keyup');
			$('#sound_test').hide();
			$('#header').show();
			socket.emit('handshake', {subject_id});
		}
	});
	$('#sound_test_button').click(function() {
		test_sound.play();
		$('#test_sound_input').focus();
	});
	$('#sound_test').show();
});
