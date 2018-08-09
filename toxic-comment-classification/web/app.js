import axios from 'axios';
import {
  Chart
} from 'chart.js';

import debounce from 'debounce';

const labels = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate'];

const unlikelyToxicColor = '#27bef9';
const likelyToxicColor = '#6200ea';
let charts = [];

const predict = text => {
  return axios.post(
    'http://localhost:5000/predict', {
      text
    }
  ).then(response => response.data);
}

const delay = 250;
const predictAndUpdateWithDelay = debounce(event => {
  if (!event.target.value) {
    location.reload();
    return;
  }

  document.querySelectorAll('.spinner').forEach($spinner => $spinner.style.display = 'block');
  document.querySelector('.overview article form div').style.display = 'none';

  predict(event.target.value).then(results => {

    document.querySelectorAll('.overview article').forEach($article => $article.style.visibility = 'visible');

    updateTextualResults(results);
    updateWordImportances(results.word_importances);

    if (charts.length) {
      charts.forEach(chart => chart.destroy());
      charts = [];
    }

    updateChart('overall-chart', results.models_averaged_probabilities);
    updateChart('resnet-1-chart', results.probabilities_of_models[0]);
    updateChart('resnet-2-chart', results.probabilities_of_models[1]);
    document.querySelectorAll('.results').forEach($result => $result.style.visibility = 'visible');

  });
  document.querySelector('.overview article form div').style.display = 'block';
  document.querySelectorAll('.spinner').forEach($spinner => $spinner.style.display = 'none');
}, delay);

document.querySelector('textarea').addEventListener('input', predictAndUpdateWithDelay);

const updateTextualResults = results => {
  const resnet64Prediction = results.probabilities_of_models_with_labels[0].sort((lhs, rhs) => rhs.probability - lhs.probability)[0];
  const resnet128Prediction = results.probabilities_of_models_with_labels[1].sort((lhs, rhs) => rhs.probability - lhs.probability)[0];

  document.querySelector('.dot').style.backgroundColor = likelyToxicColor;
  document.querySelector('form div strong').innerHTML = `Likely ${results.most_probable_category.label}.`;
  document.querySelector('form div h3').innerHTML = `
    <small>
      <strong>${ (results.most_probable_category.probability.toFixed(5) * 100).toFixed(2) } %</strong>
    </small> likely <strong> ${results.most_probable_category.label}</strong>.
  `;
  document.querySelectorAll('.results h3')[0].innerHTML = `
    <small> This model thinks the text is 
      <strong>${ (resnet64Prediction.probability.toFixed(5) * 100).toFixed(2) } %</strong>
    </small> likely <strong> ${resnet64Prediction.label}</strong>.
  `;
  document.querySelectorAll('.results h3')[1].innerHTML = `
  <small> This model thinks the text is 
    <strong>${ (resnet128Prediction.probability.toFixed(5) * 100).toFixed(2) } %</strong>
  </small> likely <strong> ${resnet128Prediction.label}</strong>.
  `;
  if (results.most_probable_category.probability <= 0.5) {
    document.querySelector('.dot').style.backgroundColor = unlikelyToxicColor;
    document.querySelector('form div strong').innerHTML = `Unlikely ${results.most_probable_category.label}.`;
  }
};

const updateWordImportances = (importances) => {
  document.querySelector('article form ul').innerHTML =
    importances
    .sort((lhs, rhs) => rhs.importance - lhs.importance)
    .slice(0, 10)
    .map(
      (importance, index) => `
      <li>
        <div style="background-color:rgba(98, 0, 234, ${ 1.0 / (index + 2)})">
          <div>${importance.word}</div>
          <div>${importance.importance.toFixed(5)}</div>
        </div>
      </li>
      `
    ).join('');
}

const updateChart = (chartId, results) => {
  document.getElementById(chartId).style.display = 'block';
  const context = document.getElementById(chartId).getContext('2d');
  const chart = new Chart(context, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Toxicity Probability',
        data: results.map(number => number.toFixed(5) * 100),
        backgroundColor: Array(results.length).fill('rgba(98, 0, 234, 0.2)'),
        borderColor: Array(results.length).fill('rgba(98, 0, 234, 1)'),
        borderWidth: 1.5
      }]
    },
    options: chartOptions
  });
  charts.push(chart);
};

const chartOptions = {
  scales: {
    yAxes: [{
      ticks: {
        beginAtZero: true,
        max: 100,
        steps: 50
      }
    }]
  }
};