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

const bonus_audio = [
	new Audio('sounds/bonus0.m4a'),
	new Audio('sounds/bonus1.m4a'),
	new Audio('sounds/bonus2.m4a'),
];

const catch_instruction = new Audio('sounds/catch_instruction.mp4');

const message_sound = new Audio('sounds/message_sound.m4a');

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

let array = [];
function preloadArray(object_array) {
	array = object_array;
	for (let i=0; i < object_array.length; i++) {
		$(`#object_array_${i}`).attr('src', `images/shapes/${object_array[i]}.png`);
	}
}

function showArray() {
	$('#object_array').show();
}

function hideArray() {
	$('#object_array').hide();
	$('img[id^="object_array_"]').css('opacity', '1');
}

function showWord(word, bubble=null) {
	if (bubble)
		$('#word').css('background-image', `url(${bubble})`);
	$('#word').html(word);
	$('#word').show();
}

function playWord(sound) {
	word_audio[sound].play();
}

function hideWord() {
	$('#word').hide();
	$('#word').css('background-image', '');
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
	if (label.match(/(amar|anar|aran|azu|azz|bk|bl|cast|cinz|gelb|gial|gr|jal|jaun|lar|lil|negr|ner|noir|oran|pard|pin|pn|pk|pret|purp|red|roj|ros|rot|roug|rox|rubr|schw|verd|verm|vert|viol|weis|whit|wit|yel|yl|yw|yo)/))
		return false;
	if (expected_label) {
		if (label.slice(0, 3) != expected_label.slice(0, 3))
			return false;
	}
	return true;
}

const span_red = '<span style="color: red; text-decoration: line-through;">';
const span_green = '<span style="color: green; font-weight:bold;">';
const span_close = '</span>';
function diffLabels(s1, s2) {
	if (s1 === s2)
		return s1;
	const sequence_matchcer = new difflib.SequenceMatcher(null, s1, s2);
	let feedback = '';
	for (let [tag, i1, i2, j1, j2] of sequence_matchcer.getOpcodes()) {
		if (tag === 'equal') {
			feedback += s1.slice(i1, i2);
		} else if (tag === 'delete') {
			feedback += span_red + s1.slice(i1, i2) + span_close;
		} else if (tag === 'insert') {
			feedback += span_green + s2.slice(j1, j2) + span_close;
		} else if (tag === 'replace') {
			feedback += span_red + s1.slice(i1, i2) + span_close;
			feedback += span_green + s2.slice(j1, j2) + span_close;
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
	$('#consent_session_time').html(payload.session_time);
	$('#consent_basic_pay').html('£' + (payload.basic_pay/100).toFixed(2));
	$('#consent_max_pay').html('£' + (payload.max_pay/100).toFixed(2));
	socket.emit('next', {subject_id, 'initialization': true});
});

socket.on('consent', function(payload) {
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
	updateBonus(payload.total_bonus);
	$('#start_training').click(function() {
		$('#start_training').off('click');
		$('#start_training').hide();
		$('#training_instructions').hide();
		socket.emit('ready_to_assign', {subject_id});
	});
	setTimeout(function() {
		enableButton('#start_training');
	}, payload.instruction_time);
	disableButton('#start_training');
	$('#training_instructions').show();
	$('#start_training').show();
	$('#header').show();
	$('#experiment').show();
});

socket.on('test_instructions', function(payload) {
	updateBonus(payload.total_bonus);
	updateProgress(payload.progress);
	$('#start_test').click(function() {
		$('#start_test').off('click');
		$('#start_test').hide();
		socket.emit('next', {subject_id});
	});
	setTimeout(function() {
		enableButton('#start_test');
	}, payload.instruction_time);
	disableButton('#start_test');
	$('#test_instructions').show();
	$('#start_test').show();
	$('#header').show();
	$('#experiment').show();
});

socket.on('comm_instructions', function(payload) {
	updateBonus(payload.total_bonus);
	updateProgress(payload.progress);
	$('#start_test').click(function() {
		$('#start_test').off('click');
		$('#start_test').hide();
		$('#spinner').show();
		setTimeout(function() {
			socket.emit('next_communication', {subject_id});
		}, 100);
	});
	setTimeout(function() {
		enableButton('#start_test');
	}, payload.instruction_time);
	disableButton('#start_test');
	$('#comm_instructions').show();
	$('#start_test').show();
	$('#header').show();
	$('#experiment').show();
});

socket.on('training_block', function(payload) {
	updateBonus(payload.total_bonus);
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
						timer.stop();
						$('#timer_bar').hide();
						$('#timer').css('width', '998px');
						setTimeout(function() {
							// 4. After 2*pause_time, hide the word and object buttons and request the next trial
							hideObject();
							hideWord();
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
						if (response_time < payload.test_trial.max_response_time) {
							if (label === payload.test_trial.word) {
								bonus_audio[2].play();
								updateBonus(payload.total_bonus_with_full);
							} else if (label.substr(0, 3) === payload.test_trial.word.substr(0, 3)) {
								bonus_audio[1].play();
								updateBonus(payload.total_bonus_with_part);
							} else {
								bonus_audio[0].play();
								updateBonus(payload.total_bonus);
							}
						}
					} else {
						showInputError('#label');
					}
					return false;
				});
				// 2. After pause_time, show the object and input box
				showObject();
				$('#timer').css('width', '998px');
				$('#timer_bar').show();
				const timer = $('#timer').animate({width: 0}, payload.test_trial.max_response_time, 'linear');
				const start_time = performance.now();
				showLabelInput();
				if (payload.test_trial.catch_trial)
					catch_instruction.play();
			}, payload.pause_time);
		}
	);
	$('#experiment').show();
	$('#header').show();
});

socket.on('test_production', function(payload) {
	$('#test_instructions').hide();
	updateBonus(payload.total_bonus);
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

				////////////////////////////////////
				// RICH FEEDBACK
				showWord(diffLabels(label, payload.word));
				////////////////////////////////////

				if (label === payload.word) {
					bonus_audio[2].play();
					updateBonus(payload.total_bonus_with_full);
				} else {
					updateBonus(payload.total_bonus);
					bonus_audio[0].play();
				}
				setTimeout(function() {
					hideObject();
					socket.emit('next', {subject_id, response: {
						test_type: 'test_production',
						shape: payload.shape,
						color: payload.color,
						expected_label: payload.word,
						input_label: label,
						response_time,
					}});
				}, payload.pause_time * 2);
			} else {
				playWord(payload.shape);
				showInputError('#label');
			}
			return false;
		});
		// 2. After pause_time, show the object and input box
		playWord(payload.shape);
		showObject();
		showLabelInput();
		const start_time = performance.now();
	}, payload.pause_time);
	$('#experiment').show();
	$('#header').show();
});

socket.on('test_comprehension', function(payload) {
	$('#test_instructions').hide();
	updateBonus(payload.total_bonus);
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

			////////////////////////////////////
			// RICH FEEDBACK
			for (let i in array) {
				if (!payload.items.includes(array[i]))
					$(`#object_array_${i}`).css('opacity', '0.1');
			}
			////////////////////////////////////

			// 3. object clicked, hide array and move on
			if (payload.items.includes(selected_item)) {
				bonus_audio[2].play();
				updateBonus(payload.total_bonus_with_full);
			} else {
				bonus_audio[0].play();
				updateBonus(payload.total_bonus);
			}
			setTimeout(function() {
				hideArray();
				socket.emit('next', {subject_id, response: {
					test_type: 'test_comprehension',
					word: payload.word,
					items: payload.items,
					array: payload.array,
					selected_button,
					selected_item,
					response_time,
				}});
			}, payload.pause_time * 2);
		}).css('cursor', 'pointer');
		// 2. After pause_time, show the object and input box
		showWord(payload.word);
		showArray();
		const start_time = performance.now();
	}, payload.pause_time);
	$('#experiment').show();
	$('#header').show();
});

socket.on('comm_production', function(payload) {
	$('#comm_instructions').hide();
	$('#spinner').hide();
	updateBonus(payload.total_bonus);
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
				showWord(label, 'images/bubble_prod.png');
				$('#feedback_object').attr('src', 'images/waiting_comp.gif').show();
				socket.emit('send_label', {subject_id, response: {
					test_type: 'comm_production',
					shape: payload.shape,
					color: payload.color,
					item: payload.item,
					expected_label: payload.word,
					input_label: label,
					response_time,
				}});
			} else {
				playWord(payload.shape);
				showInputError('#label');
			}
			return false;
		});
		// 2. After pause_time, show the object and input box
		playWord(payload.shape);
		showObject();
		showLabelInput();
		const start_time = performance.now();
	}, payload.pause_time);
	$('#experiment').show();
	$('#header').show();
});

socket.on('comm_comprehension', function(payload) {
	$('#comm_instructions').hide();
	updateBonus(payload.total_bonus);
	updateProgress(payload.progress);
	hideArray();
	hideWord();
	preloadArray(payload.array);
	showArray();
	$('#spinner').show();
	$('#experiment').show();
	$('#header').show();
});

socket.on('receive_message', function(payload) {
	$('#spinner').hide();
	$('img[id^="object_array_"]').click(function() {
		$('img[id^="object_array_"]').off('click');
		$('img[id^="object_array_"]').css('cursor', 'default');
		const response_time = Math.floor(performance.now() - start_time);
		const selected_button = parseInt($(this).attr('id').match(/object_array_(.+)/)[1]);
		const selected_item = array[selected_button];
		const correct_object_position = array.indexOf(payload.item);
		for (let i in array) {
			if (i != correct_object_position)
				$(`#object_array_${i}`).css('opacity', '0.1');
		}
		if (selected_item === payload.item) {
			bonus_audio[2].play();
			updateBonus(payload.total_bonus_with_full);
		} else {
			bonus_audio[0].play();
		}
		setTimeout(function() {
			hideArray();
			socket.emit('next_communication', {subject_id});
		}, payload.pause_time * 2);
		socket.emit('send_feedback', {subject_id, response: {
			test_type: 'comm_comprehension',
			item: payload.item,
			word: payload.label,
			selected_button,
			selected_item,
			response_time,
		}});
	}).css('cursor', 'pointer');
	message_sound.play();
	showWord(payload.label, 'images/bubble_comp.png');
	const start_time = performance.now();
});

socket.on('receive_feedback', function(payload) {
	const [shape, color] = payload.selected_item.split('_');
	$('#feedback_object').attr('src', `images/shapes/${shape}_${color}.png`);
	updateBonus(payload.total_bonus);
	if (payload.selected_item === payload.target_item)
		bonus_audio[2].play();
	else
		bonus_audio[0].play();
	setTimeout(function() {
		hideObject();
		$('#feedback_object').hide();
		socket.emit('next_communication', {subject_id});
	}, payload.pause_time * 2);
});

socket.on('questionnaire', function(payload) {
	updateBonus(payload.total_bonus);
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
	$('#header').show();
	$('#comments').focus();
});

socket.on('end_of_experiment', function(payload) {
	updateBonus(payload.total_bonus);
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
	$('#header').show();
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
		if ($(this).val().toLowerCase() === test_phrase || $(this).val() === '/') {
			$('#test_sound_input').off('keyup');
			$('#sound_test').hide();
			socket.emit('handshake', {subject_id});
		}
	});
	$('#sound_test_button').click(function() {
		test_sound.play();
		$('#test_sound_input').focus();
	});
	$('#sound_test').show();
});
